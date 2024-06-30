from typing import (
    TYPE_CHECKING,
)

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import (
        EntryArchive,
    )
    from structlog.stdlib import (
        BoundLogger,
    )

from nomad.config import config
from nomad.datamodel.results import Material, Results
from nomad.parsing.parser import MatchingParser
from nomad.parsing.file_parser import TextParser, Quantity
from nomad.units import ureg

import numpy as np

configuration = config.get_plugin_entry_point('nomad_parser_fhiaims.parsers:myparser')


class Quantity(Quantity):
    """
    Class to define a table of quantities to be parsed in the TextParser.

    Extends the semantics of repeats to accept, next to `bool` and `int`, a `dict` mapping.

    Arguments:
        quantity: string to identify the name or a metainfo quantity to initialize the
            quantity object.
        re_pattern: pattern to be used by re for matching. Ideally, overlaps among
            quantities for a given parser should be avoided.
        group_names: mapping from match group indices (-1) to their corresponding names.
    """

    def __init__(
        self,
        quantity: Union[str, 'mQuantity'],
        re_pattern: Union[str, ParsePattern],
        group_names: Dict[int, str] = None,
        **kwargs,
    ):
        super().__init__(quantity, re_pattern, **kwargs)
        self.group_names = group_names or {}

    def parse(self, text: str) -> List[Quantity]:
        """
        Parse the text to extract quantities based on the re_pattern and group_names.

        Arguments:
            text: The text to parse.

        Returns:
            List of Quantity objects.
        """
        quantities = []
        pattern = re.compile(self.re_pattern)
        matches = pattern.finditer(text)

        if isinstance(self.repeats, dict):
            for match in matches:
                row = {}
                for group_idx, name in self.repeats.items():
                    try:
                        value = match.group(group_idx)
                        if value:
                            quantity = Quantity(name, self.re_pattern, **self.kwargs)
                            quantity_data = quantity.to_data(value)
                            row[name] = quantity_data
                    except IndexError:
                        # group index not found in the match, skip
                        continue
                quantities.append(row)
        else:
            for match in matches:
                for group_idx, name in self.group_names.items():
                    try:
                        value = match.group(group_idx)
                        if value:
                            quantity = Quantity(name, self.re_pattern, **self.kwargs)
                            quantity_data = quantity.to_data(value)
                            quantities.append(Quantity(name, value, **self.kwargs))
                    except IndexError:
                        # group index not found in the match, skip
                        continue

        return quantities


class FHIAimsOutParser(TextParser):
    def __init__(self):
        # TODO move these to text parser?
        super().__init__(None)

        self.re_non_greedy = r'[\s\S]+?'
        self.re_blank_line = r'^\s*$'
        self.re_float = r'[\-+]?\d+\.?\d*[Ee]?[\-+]?\d*'
        self.re_n = r'[\n\r]'  # ? make it eol with `$`

        self.re_sep_short = r'\-{60}$'
        self.re_sep_long = r'\-{78}$'

    def capture(to_match, rep: int = 0):
        if rep:
            return rf'[\s+({to_match})]{{{rep}}}'
        return rf'({to_match})'

    def init_quantities(self):
        # new quantities
        geometry_description = [
            Quantity(
                'symmetry',
                r'Symmetry information'
                + self.capture(self.re_non_greedy)
                + self.re_blank_line,
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'precision',
                            r'Precision set to\s+' + self.capture({self.re_float}),
                            dtype=np.float64,
                        ),
                        Quantity(
                            'space_group_number',
                            r'Space group\s+: ' + self.capture(r'\d+'),
                            dtype=int,
                        ),
                        Quantity(
                            'space_group_symbol',
                            r'International\s+: ' + self.capture(r'[\w\d]+'),
                            dtype=str,
                        ),
                        Quantity(
                            'space_group_schoenflies',
                            r'Schoenflies\s+: ' + self.capture(r'[\w\d]+'),
                            dtype=str,
                        ),
                    ]
                ),
            ),
            Quantity(
                'geometry',
                r'Input geometry:'
                + self.capture(self.re_non_greedy)
                + self.re_blank_line,
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'unit_cell',
                            r'Unit cell:'
                            + self.capture(self.re_non_greedy)
                            + r'Atomic structure:'
                            + self.capture(self.re_non_greedy)
                            + self.re_blank_line,
                            sub_parser=TextParser(
                                quantities=[
                                    Quantity(
                                        'lattice_vector',
                                        r'\s+\|'
                                        + self.capture(self.re_non_greedy)
                                        + r'$',
                                        repeats=True,
                                        sub_parser=TextParser(
                                            quantities=[
                                                Quantity(
                                                    'vector',
                                                    self.capture(self.re_float, rep=3),
                                                    dtype=np.float64,
                                                    repeats=3,
                                                ),
                                            ]
                                        ),
                                    ),
                                ]
                            ),
                        ),
                        Quantity(
                            'atomic_structure',
                            r'\d+: Species([\w\s]+)',
                            repeats=True,
                            sub_parser=TextParser(
                                quantities=[
                                    Quantity(
                                        'species',
                                        r'([A-Z][a-z]*)',
                                        repeats=True,
                                    ),
                                    Quantity(
                                        'cartesian_positions',
                                        self.re_float,
                                        repeats=3,
                                        dtype=np.float64,
                                    ),
                                ]
                            ),
                        ),
                    ]
                ),
            ),
            Quantity(
                'lattice_parameters',
                r'Lattice parameters for 3D lattice'
                + self.capture(self.re_non_greedy)
                + r'$',
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'lattice_units',
                            r'\(in ' + self.capture(r'[\w]+') + r'\)',
                            str_operation=lambda x: ureg(x),  #! map
                        ),
                        Quantity(
                            'lattice_vectors',
                            self.capture(self.re_float, rep=3),
                            dtype=np.float64,
                            repeats=3,
                        ),
                    ]
                ),
            ),
            Quantity(
                'lattice_angles',
                r'Angle(s) between unit vectors' + self.re_non_greedy + r'$',
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'lattice_units',
                            r'\(in ' + self.capture(r'[\w]+') + r'\)',
                            str_operation=lambda x: ureg(x),  #! map
                        ),
                        Quantity(
                            'lattice_angles',
                            self.capture(self.re_float, rep=3),
                            dtype=np.float64,
                            repeats=3,
                        ),
                    ]
                ),
            ),
            Quantity(
                'lattice_derived',
                r'Quantities derived from the lattice vectors:'
                + self.capture(self.re_non_greedy)
                + self.re_blank_line,
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'reciprocal_lattice_vector',
                            r'Reciprocal lattice vector'
                            + self.capture(self.re_non_greedy)
                            + r'$',
                            repeats=True,
                            sub_parser=TextParser(
                                quantities=[
                                    Quantity(
                                        'vector',
                                        self.capture(self.re_float, rep=3),
                                        dtype=np.float64,
                                        repeats=3,
                                    ),
                                ]
                            ),
                        ),
                        Quantity(
                            'cell_volume',
                            r'Unit cell volume\s+:'
                            + self.capture(self.re_float)
                            + r'\w+',
                            dtype=np.float64,
                        ),
                        Quantity(
                            'cell_volume_units',
                            r'Unit cell volume\s+:'
                            + self.re_float
                            + self.capture(r'\w+'),
                            str_operation=lambda x: ureg(x),  #! map
                        ),
                    ]
                ),
            ),
        ]

        scf_output = [
            Quantity(
                'eigenvalues',
                r'Writing Kohn\-Sham eigenvalues.'
                + self.re_blank_line
                + self.re_non_greedy
                + self.re_blank_line,
                repeats=True,
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'line',
                            self.capture(self.re_float, rep=4),
                            repeats={1: 'state', 2: 'occupation', 4: 'eigenvalue'},
                        ),
                    ]
                ),
            ),
            Quantity(
                'energy_components',  # format 1
                r'Total energy components:'
                + self.capture(self.re_non_greedy)
                + self.re_blank_line,
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'eigenvalues',
                            r'Sum of eigenvalues'
                            + self.re_non_greedy
                            + self.capture(self.re_float)
                            + r' eV$',
                            dtype=np.float64,
                        ),
                        Quantity(
                            'xc_energy',
                            r'XC energy correction'
                            + self.re_non_greedy
                            + self.capture(self.re_float)
                            + r' eV$',
                            dtype=np.float64,
                        ),
                        Quantity(
                            'xc_potential',
                            r'XC potential correction'
                            + self.re_non_greedy
                            + self.capture(self.re_float)
                            + r' eV$',
                            dtype=np.float64,
                        ),
                        Quantity(
                            'atomic_electrostatic',
                            r'Free-atom electrostatic energy'
                            + self.re_non_greedy
                            + self.capture(self.re_float)
                            + r' eV$',
                            dtype=np.float64,
                        ),
                        Quantity(
                            'hartree',
                            r'Hartree energy correction'
                            + self.re_non_greedy
                            + self.capture(self.re_float)
                            + r' eV$',
                            dtype=np.float64,
                        ),
                        Quantity(
                            'entropy',
                            r'Entropy correction'
                            + self.re_non_greedy
                            + self.capture(self.re_float)
                            + r' eV$',
                            dtype=np.float64,
                        ),
                        Quantity(
                            'total_energy',
                            r'Total energy'
                            + self.re_non_greedy
                            + self.capture(self.re_float)
                            + r' eV$',
                            dtype=np.float64,
                        ),
                        Quantity(
                            'total_energy_extrapolation',
                            r'Total energy, T -> 0'
                            + self.re_non_greedy
                            + self.capture(self.re_float)
                            + r' eV',
                            dtype=np.float64,
                        ),
                        Quantity(
                            'free_energy',
                            r'Electronic free energy'
                            + self.re_non_greedy
                            + self.capture(self.re_float)
                            + r' eV$',
                            dtype=np.float64,
                        ),
                    ]
                ),
            ),
            Quantity(
                'scf_convergence',  # format 2
                r'SCF\s\d+ :([\S\s]+)$',  # ! handle warnings
                split_on=r'\|',
                repeats={
                    2: 'density',
                    3: 'eigen_energy',
                    4: 'total_energy',
                    5: 'total_forces',
                },
            ),
        ]

        ion_output = [
            Quantity(
                'energy_components',
                r'Start decomposition of the XC Energy'
                + self.capture(self.re_non_greedy)
                + r'End decomposition of the XC Energy',
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'hartree',
                            r'Hartree-Fock part'
                            + self.re_non_greedy
                            + self.capture(self.re_float)
                            + r' eV$',
                            dtype=np.float64,
                        ),
                        Quantity(
                            'x',
                            r'X Energy'
                            + self.re_non_greedy
                            + self.capture(self.re_float)
                            + r' eV$',
                            dtype=np.float64,
                        ),
                        Quantity(
                            'gga_c',
                            r'C Energy GGA'
                            + self.re_non_greedy
                            + self.capture(self.re_float)
                            + r' eV$',
                            dtype=np.float64,
                        ),
                        Quantity(
                            'xc_energy',
                            r'Total XC Energy'
                            + self.re_non_greedy
                            + self.capture(self.re_float)
                            + r' eV$',
                            dtype=np.float64,
                        ),
                        Quantity(
                            'lda_x',
                            r'X Energy LDA'
                            + self.re_non_greedy
                            + self.capture(self.re_float)
                            + r' eV$',
                            dtype=np.float64,
                        ),
                        Quantity(
                            'lda_c',
                            r'C Energy LDA'
                            + self.re_non_greedy
                            + self.capture(self.re_float)
                            + r' eV$',
                            dtype=np.float64,
                        ),
                    ]
                ),
            ),
        ]

        workflows = [
            Quantity(
                Program.version,
                r'(?:Version|FHI\-aims version)\s*\:*\s*([\d\.]+)\s*',
                repeats=False,
            ),
            Quantity(
                'compilation',
                r'Compiled on (\d+) at (\d+\:\d+\:\d+) on host (\S+)\.$',
                repeats={
                    1: 'date',
                    2: 'time',
                    3: Program.compilation_host,
                },  # modify this to use Quantity, but shallow
            ),
            Quantity(
                'ion_step',
                r'Begin self-consistency loop:'
                + self.capture(self.re_non_greedy)
                + r'Writing the current geometry to file|'
                + self.re_sep_long,
                repeats=True,
                sub_parser=TextParser(
                    quantities=[
                        ion_output,
                        Quantity(
                            'scf_step',
                            r'Begin self-consistency iteration|Convergence:'
                            + self.capture(self.re_non_greedy)
                            + self.re_sep_short
                            + self.re_blank_line
                            + self.re_sep_short,
                            repeats=True,
                            sub_parser=TextParser(
                                quantities=[geometry_description, scf_output]
                            ),
                        ),
                    ]
                ),
            ),
        ]

        # old quantities

        calculation_quantities = [
            Quantity(
                'structure',
                rf'Atomic structure(.|\n)*\| *Atom *x \[A\] *y \[A\] *z \[A\]([\s\S]+?Species[\s\S]+?(?:{self.re_n} *{self.re_n}| 1\: ))',
                repeats=False,
                convert=False,
                sub_parser=TextParser(quantities=structure_quantities),
            ),
            Quantity(
                'structure',
                rf'{self.re_n} *(atom +{self.re_float}[\s\S]+?(?:{self.re_n} *{self.re_n}|\-\-\-))',
                repeats=False,
                convert=False,
                sub_parser=TextParser(quantities=structure_quantities),
            ),
            Quantity(  # This quantity is double defined in self._quantities
                'lattice_vectors',
                rf'{self.re_n} *lattice_vector([\d\.\- ]+){self.re_n} *lattice_vector([\d\.\- ]+){self.re_n} *lattice_vector([\d\.\- ]+)',
                unit='angstrom',
                repeats=False,
                shape=(3, 3),
                dtype=float,
            ),
            Quantity(
                'energy',
                rf'{self.re_n} *Energy and forces in a compact form:([\s\S]+?(?:{self.re_n}{self.re_n}|Electronic free energy\s*:\s*[\d\.\-Ee]+ eV))',
                str_operation=str_to_energy_components,
                repeats=False,
                convert=False,
            ),
            # in some cases, the energy components are also printed for after a calculation
            # same format as in scf iteration, they are printed also in initialization
            # so we should get last occurence
            Quantity(
                'forces',
                rf'{self.re_n} *Total atomic forces.*?\[eV/Ang\]:\s*([\d\.Ee\-\+\s\|]+)',
                str_operation=str_to_atomic_forces,
                repeats=False,
                convert=False,
            ),
            # TODO no metainfo for scf forces but old parser put it in atom_forces_free_raw
            Quantity(
                'forces_raw',
                rf'{self.re_n} *Total forces\([\s\d]+\)\s*:([\s\d\.\-\+Ee]+){self.re_n}',
                repeats=True,
                dtype=float,
            ),
            Quantity(
                'time_calculation',
                rf'{self.re_n} *\| Time for this force evaluation\s*:\s*[\d\.]+ s\s*([\d\.]+) s',
                repeats=False,
                dtype=float,
            ),
            Quantity(
                'total_dos_files',
                r'Calculating total density of states([\s\S]+?)\-{5}',
                str_operation=str_to_dos_files,
                repeats=False,
                convert=False,
            ),
            Quantity(
                'atom_projected_dos_files',
                r'Calculating atom\-projected density of states([\s\S]+?)\-{5}',
                str_operation=str_to_dos_files,
                repeats=False,
                convert=False,
            ),
            Quantity(
                'species_projected_dos_files',
                r'Calculating angular momentum projected density of states([\s\S]+?)\-{5}',
                str_operation=str_to_dos_files,
                repeats=False,
                convert=False,
            ),
            Quantity(
                'vdW_TS',
                rf'(Evaluating non\-empirical van der Waals correction[\s\S]+?)(?:\|\s*Converged\.|\-{5}{self.re_n}{self.re_n})',
                repeats=False,
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'kind',
                            r'Evaluating non\-empirical van der Waals correction \(([\w /]+)\)',
                            repeats=False,
                            convert=False,
                            flatten=False,
                        ),
                        Quantity(
                            'atom_hirshfeld',
                            r'\| Atom\s*\d+:([\s\S]+?)\-{5}',
                            str_operation=str_to_hirshfeld,
                            repeats=True,
                            convert=False,
                        ),
                    ]
                ),
            ),
            Quantity(
                'converged',
                r'Self\-consistency cycle (converged)\.',
                repeats=False,
                dtype=str,
            ),
            date_time,
        ]

        molecular_dynamics_quantities = [
            Quantity(
                'md_run',
                r' *Running\s*Born-Oppenheimer\s*molecular\s*dynamics\s*in\s*([A-Z]{3})\s*ensemble*\D*\s*with\s*([A-Za-z\-]*)\s*thermostat',
                repeats=False,
                convert=False,
            ),
            Quantity(
                'md_timestep',
                rf'{self.re_n} *Molecular dynamics time step\s*=\s*({self.re_float} [A-Za-z]*)\s*{self.re_n}',
                str_operation=str_to_quantity,
                repeats=False,
                convert=False,
            ),
            Quantity(
                'md_simulation_time',
                rf'{self.re_n} *\| *simulation time\s*=\s*({self.re_float} [A-Za-z]*)\s*{self.re_n}',
                str_operation=str_to_quantity,
                repeats=False,
                convert=False,
            ),
            Quantity(
                'md_temperature',
                rf'{self.re_n} *\| *at temperature\s*=\s*({self.re_float} [A-Za-z]*)\s*{self.re_n}',
                str_operation=str_to_quantity,
                repeats=False,
                convert=False,
            ),
            Quantity(
                'md_thermostat_mass',
                rf'{self.re_n} *\| *thermostat effective mass\s*=\s*({self.re_float})\s*{self.re_n}',
                str_operation=str_to_quantity,
                repeats=False,
                convert=False,
            ),
            Quantity(
                'md_thermostat_units',
                rf'Thermostat\s*units\s*for\s*molecular\s*dynamics\s*:\s*([A-Za-z\^\-0-9]*)',
                str_operation=str_to_ureg,
                repeats=False,
                convert=False,
            ),
            Quantity(
                'md_calculation_info',
                rf'{self.re_n} *Advancing structure using Born-Oppenheimer Molecular Dynamics:\s*{self.re_n}'
                rf' *Complete information for previous time-step:'
                rf'([\s\S]+?)((?:{self.re_n}{self.re_n}|\| Nose-Hoover Hamiltonian\s*:\s*[Ee\d\.\-\+]+ eV))',
                str_operation=str_to_md_calculation_info,
                repeats=False,
                convert=False,
            ),
            Quantity(
                'md_system_info',
                rf'Atomic structure.*as used in the preceding time step:\s*{self.re_n}'
                rf'([\s\S]+?)((?:{self.re_n}{self.re_n}|\s*Begin self-consistency loop))',
                repeats=False,
                convert=False,
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'positions',
                            rf'atom +({self.re_float})\s+({self.re_float})\s+({self.re_float})',
                            dtype=np.dtype(np.float64),
                            repeats=True,
                        ),
                        Quantity(
                            'velocities',
                            rf'velocity\s+({self.re_float})\s+({self.re_float})\s+({self.re_float})',
                            dtype=np.dtype(np.float64),
                            repeats=True,
                        ),
                    ]
                ),
            ),
        ]

        tail = '|'.join(
            [
                r'Time for this force evaluation\s*:\s*[s \d\.]+',
                r'Final output of selected total energy values',
                r'No geometry change',
                r'Leaving FHI\-aims',
                r'\Z',
            ]
        )

        self._quantities = [
            Quantity(
                TimeRun.wall_start,
                r'Internal wall clock time zero\s*:\s*([0-9\-E\.]+)\s*(?P<__unit>\w+)\.',
                repeats=False,
                dtype=float,
            ),
            Quantity(Run.raw_id, r'aims_uuid\s*:\s*([\w\-]+)', repeats=False),
            Quantity(
                xsection_run.x_fhi_aims_number_of_tasks,
                r'Using\s*(\d+)\s*parallel tasks',
                repeats=False,
            ),
            Quantity(
                x_fhi_aims_section_parallel_task_assignement.x_fhi_aims_parallel_task_nr,
                r'Task\s*(\d+)\s*on host',
                repeats=True,
            ),
            Quantity(
                x_fhi_aims_section_parallel_task_assignement.x_fhi_aims_parallel_task_host,
                r'Task\s*\d+\s*on host\s*([\s\S]+?)reporting',
                repeats=True,
                flatten=False,
            ),
            Quantity(
                'fhi_aims_files',
                r'(?:FHI\-aims file:|Parsing)\s*([\w\/\.]+)',
                repeats=True,
            ),
            Quantity(
                'array_size_parameters',
                r'Basic array size parameters:\s*([\|:\s\w\.\/]+:\s*\d+)',
                repeats=False,
                str_operation=str_to_array_size_parameters,
                convert=False,
            ),
            Quantity(
                xsection_method.x_fhi_aims_controlInOut_hse_unit,
                r'hse_unit: Unit for the HSE06 hybrid functional screening parameter set to\s*(\w)',
                str_operation=FHIAimsControlParser.str_to_unit,
                repeats=False,
            ),
            Quantity(
                xsection_method.x_fhi_aims_controlInOut_hybrid_xc_coeff,
                r'hybrid_xc_coeff: Mixing coefficient for hybrid-functional exact exchange modified to\s*([\d\.]+)',
                repeats=False,
                dtype=float,
            ),
            Quantity(
                'k_grid', rf'{self.re_n} *Found k-point grid:\s*([\d ]+)', repeats=False
            ),  # taken from tests/data/fhi_aims
            Quantity(
                xsection_run.x_fhi_aims_controlInOut_MD_time_step,
                rf'{self.re_n} *Molecular dynamics time step\s*=\s*([\d\.]+)\s*(?P<__unit>[\w]+)',
                repeats=False,
            ),
            Quantity(
                xsection_method.x_fhi_aims_controlInOut_relativistic,
                rf'{self.re_n} *Scalar relativistic treatment of kinetic energy:\s*([\w\- ]+)',
                repeats=False,
            ),
            Quantity(
                xsection_method.x_fhi_aims_controlInOut_relativistic,
                rf'{self.re_n} *(Non-relativistic) treatment of kinetic energy',
                repeats=False,
            ),
            Quantity(
                xsection_method.x_fhi_aims_controlInOut_relativistic_threshold,
                rf'{self.re_n} *Threshold value for ZORA:\s*([\d\.Ee\-\+])',
                repeats=False,
            ),
            Quantity(
                xsection_method.x_fhi_aims_controlInOut_xc,
                rf'{self.re_n} *XC:\s*(?:Using)*\s*([\w\- ]+) with OMEGA =\s*([\d\.Ee\-\+]+)',
                repeats=False,
                dtype=None,
            ),
            Quantity(
                'petukhov',
                rf'{self.re_n} *Fixing petukhov mixing factor to\s+(\d?\.[\d]+)',
                repeats=False,
                dtype=np.dtype(np.float64),
            ),
            Quantity(
                xsection_method.x_fhi_aims_controlInOut_xc,
                r'XC: (?:Running|Using) ([\-\w \(\) ]+)',
                repeats=False,
            ),
            Quantity(
                xsection_method.x_fhi_aims_controlInOut_xc,
                rf'{self.re_n} *(Hartree-Fock) calculation starts \.\.\.',
                repeats=False,
            ),
            Quantity(
                'band_segment_points',
                r'Plot band\s*\d+\s*\|\s*begin[ \d\.\-]+\s*\|\s*end[ \d\.\-]+\s*\|\s*number of points:\s*(\d+)',
                repeats=True,
            ),
            Quantity(
                'species',
                rf'(Reading configuration options for species [\s\S]+?)(?:{self.re_n} *Finished|{self.re_n} *{self.re_n})',
                str_operation=str_to_species_in,
                repeats=False,
            ),
            Quantity(
                'control_inout',
                rf'{self.re_n} *Reading file control\.in\.\s*\-*\s*([\s\S]+?)'
                r'(?:Finished reading input file \'control\.in\'|Input file control\.in ends\.)',
                repeats=False,
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'species',
                            r'Reading configuration options for (species[\s\S]+?)grid points\.',
                            repeats=True,
                            str_operation=str_to_species,
                        ),
                        *molecular_dynamics_quantities,
                    ]
                ),
            ),
            Quantity(
                'control_in_verbatim',
                rf'{self.re_n}  *Parsing control\.in([\S\s]*)Completed first pass over input file control\.in',
                repeats=False,
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'md_controlin',
                            rf' *([\_a-zA-Z\d\-]*MD[\_a-zA-Z\d\-]*)\s+([a-zA-Z\d\.\-\_\^]+.*){self.re_n}',
                            str_operation=str_to_md_control_in,
                            repeats=True,
                            convert=False,
                        )
                    ]
                ),
            ),
            # GW input quantities
            Quantity('gw_flag', self._re_gw_flag, repeats=False),
            Quantity(
                'anacon_type', rf'{self.re_n}\s*anacon_type\s*(\d+)', repeats=False
            ),
            Quantity(
                'gw_analytical_continuation',
                rf'{self.re_n} (?:Using)*\s*([\w\-\s]+) for analytical continuation',
                repeats=False,
                flatten=True,
                str_operation=lambda x: [
                    y.lower() for v in x.split(' ') for y in v.split('-')
                ],
            ),
            Quantity('k_grid', rf'{self.re_n} *k\_grid\s*([\d ]+)', repeats=False),
            Quantity(
                'freq_grid_type',
                rf'{self.re_n}\s*Initialising([\w\-\s]+)time and frequency grids',
                repeats=False,
            ),
            Quantity(
                'n_freq',
                rf'{self.re_n}\s*frequency_points\s*(\d+)',
                repeats=False,
                dtype=int,
            ),
            Quantity(
                'frequency_data',
                r'\s*\|*\s*i_freq\s*([\d*\s*.+eE\-\+]+)',
                repeats=True,
                str_operation=str_to_frequency,
            ),
            Quantity(
                'frozen_core',
                rf'{self.re_n}\s*frozen_core_scf\s*(\d+)',
                repeats=False,
                dtype=int,
            ),
            Quantity(
                'n_states_gw',
                r'\|\s*Number of Kohn-Sham states \(occupied \+ empty\)\s*\:\s*(\d+)',
                repeats=False,
            ),
            Quantity(
                'gw_self_consistency',
                r'GW Total Energy Calculation([\s\S]+?)\-{5}',
                repeats=True,
                str_operation=str_to_gw_scf,
                convert=False,
            ),
            Quantity(
                'gw_eigenvalues',
                r'(state\s*occ_num\s*e_gs[\s\S]+?)\s*\| Total time',
                str_operation=str_to_gw_eigs,
                repeats=False,
                convert=False,
            ),
            # assign the initial geometry to full scf as no change in structure is done
            # during the initial scf step
            Quantity(
                'lattice_vectors',
                r'Input geometry:\s*\|\s*Unit cell:\s*'
                r'\s*\|\s*([\d\.\-\+eE\s]+)\s*\|\s*([\d\.\-\+eE\s]+)\s*\|\s*([\d\.\-\+eE\s]+)',
                repeats=False,
                unit='angstrom',
                shape=(3, 3),
                dtype=float,
            ),
            Quantity(
                'structure',
                rf'Atomic structure(.|\n)*\| *Atom *x \[A\] *y \[A\] *z \[A\]([\s\S]+?Species[\s\S]+?(?:{self.re_n} *{self.re_n}| 1\: ))',
                repeats=False,
                convert=False,
                sub_parser=TextParser(quantities=structure_quantities),
            ),
            Quantity(
                'lattice_vectors_reciprocal',
                r'Quantities derived from the lattice vectors:\s*'
                r'\s*\|\s*Reciprocal lattice vector \d:([\d\.\-\+eE\s]+)\s*\|\s*Reciprocal lattice vector \d:([\d\.\-\+eE\s]+)\s*\|\s*Reciprocal lattice vector \d:([\d\.\-\+eE\s]+)',
                repeats=False,
                unit='1/angstrom',
                shape=(3, 3),
                dtype=float,
            ),
            Quantity(
                'full_scf',
                r'Begin self-consistency loop: Initialization' rf'([\s\S]+?(?:{tail}))',
                repeats=True,
                sub_parser=TextParser(quantities=calculation_quantities),
            ),
            Quantity(
                'geometry_optimization',
                rf'{self.re_n} *Geometry optimization: Attempting to predict improved coordinates\.'
                rf'([\s\S]+?(?:{tail}))',
                repeats=True,
                sub_parser=TextParser(quantities=calculation_quantities),
            ),
            Quantity(
                'molecular_dynamics',
                rf'{self.re_n} *Molecular dynamics: Attempting to update all nuclear coordinates\.'
                rf'([\s\S]+?(?:{tail}))',
                repeats=True,
                sub_parser=TextParser(
                    quantities=[*calculation_quantities, *molecular_dynamics_quantities]
                ),
            ),
            Quantity(
                'timing',
                r'(Date.+\s+Computational steps[\s\S]+?\Z)',
                sub_parser=TextParser(
                    quantities=[
                        date_time,
                        Quantity(
                            'total_time',
                            r'\| Total time +: +[\d\.]+ s +([\d\.]+) s',
                            dtype=np.float64,
                        ),
                    ]
                ),
            ),
        ]
        # TODO add SOC perturbed eigs, dielectric function

    def get_number_of_spin_channels(self):
        return self.get('array_size_parameters', {}).get('Number of spin channels', 1)


class MyParser(MatchingParser):
    def parse(
        self,
        mainfile: str,
        archive: 'EntryArchive',
        logger: 'BoundLogger',
        child_archives: dict[str, 'EntryArchive'] = None,
    ) -> None:
        logger.info('MyParser.parse', parameter=configuration.parameter)
        archive.results = Results(material=Material(elements=['H', 'O']))
