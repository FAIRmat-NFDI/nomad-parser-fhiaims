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


class FHIAimsOutParser(TextParser):
    def __init__(self):
        # TODO move these to text parser?
        self.re_float = r'[-+]?\d+\.\d*(?:[Ee][-+]\d+)?'
        self.re_triple_float = rf'[\s+({self.re_float})]{3}'
        self.re_n = r'[\n\r]'
        self.re_blank_line = r'^\s*$'
        self._re_gw_flag = rf'{self.re_n}\s*(?:qpe_calc|sc_self_energy)\s*([\w]+)'
        super().__init__(None)

    def init_quantities(self):
        # new quantities
        quantities = [
            Quantity(
                'symmetry',
                r'Symmetry information',
                self.re_blank_line,
                sub_parser=TextParser(
                    [
                        Quantity(
                            'precision',
                            rf'Precision set to\s+({self.re_float})',
                            dtype=np.float64,
                        ),
                        Quantity(
                            'space_group_number', r'Space group\s+: (\d+)', dtype=int
                        ),
                        Quantity(
                            'space_group_symbol',
                            r'International\s+: ([\-\w]+)',
                            dtype=str,
                        ),
                        Quantity(
                            'space_group_schoenflies',
                            r'Schoenflies\s+: ([\^\w]+)',
                            dtype=str,
                        ),
                    ]
                ),
            ),
            Quantity(
                'geometry',
                r'Input geometry:',
                self.re_blank_line,
                sub_parser=TextParser(
                    [
                        Quantity(
                            'unit_cell',
                            r'Unit cell:',
                            r'Atomic structure:',
                            sub_parser=TextParser(
                                [
                                    Quantity(
                                        'lattice_vector',
                                        r'([\d\.\-\+eE\s]+)',
                                        repeats=3,
                                        dtype=np.float64,
                                    ),
                                ]
                            ),
                        ),
                        Quantity(
                            'atomic_structure',
                            r'\d+: Species[\w\s]+',
                            repeats=True,
                            sub_parser=TextParser(
                                [
                                    Quantity(
                                        'species',
                                        r'Species ([A-Z][a-z]*)',
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
                rf'Lattice parameters for 3D lattice \(\w+\)\s+:{self.re_triple_float}',
                dtype=np.float64,
                repeats=3,
            ),
            Quantity(
                'lattice_angles',
                rf'Angle(s) between unit vectors \(\w+\)\s+:{self.re_triple_float}',
                dtype=np.float64,
                repeats=3,
            ),
            Quantity(
                'lattice_derived',
                r'Quantities derived from the lattice vectors:',
                self.re_blank_line,
                sub_parser=TextParser(
                    [
                        Quantity(
                            'reciprocal_lattice_vector',
                            rf'Reciprocal lattice vector[\s\S]+?',
                            repeats=True,
                            sub_parser=TextParser(
                                [
                                    Quantity(
                                        'vector',
                                        rf'{self.re_triple_float}',
                                        repeats=3,
                                        dtype=np.float64,
                                    ),
                                ]
                            ),
                        ),
                        Quantity(
                            'cell_volume',
                            rf'Unit cell volume\s+:{self.re_float}\w+',
                            dtype=np.float64,
                        ),
                    ]
                ),
            ),
        ]

        # old quantities

        structure_quantities = [
            Quantity(
                'labels',
                rf'(?:Species\s*([A-Z][a-z]*)|([A-Z][a-z]*)\w*{self.re_n})',
                repeats=True,
            ),
            Quantity(
                'positions',
                rf'({self.re_float})\s+({self.re_float})\s+({self.re_float}) *{self.re_n}',
                dtype=np.dtype(np.float64),
                repeats=True,
            ),
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

        eigenvalues = Quantity(
            'eigenvalues',
            rf'Writing Kohn\-Sham eigenvalues\.([\s\S]+?State[\s\S]+?)(?:{self.re_n}{self.re_n} +[A-RT-Z])',
            repeats=True,
            sub_parser=TextParser(
                quantities=[
                    Quantity(
                        'kpoints',
                        rf'{self.re_n} *K-point:\s*\d+ at\s*({self.re_float})\s*({self.re_float})\s*({self.re_float})',
                        dtype=float,
                        repeats=True,
                    ),
                    Quantity(
                        'occupation_eigenvalue',
                        rf'{self.re_n} *\d+\s*({self.re_float})\s*({self.re_float})\s*{self.re_float}',
                        repeats=True,
                    ),
                ]
            ),
        )

        date_time = Quantity(
            'date_time',
            rf'Date\s*:\s*(\d+), Time\s*:\s*([\d\.]+)\s*',
            repeats=False,
            convert=False,
            str_operation=lambda x: datetime.strptime(
                x, '%Y%m%d %H%M%S.%f'
            ).timestamp(),
        )

        scf_quantities = [
            # TODO add section_eigenvalues to scf_iteration
            date_time,
            eigenvalues,
            Quantity(
                'energy_components',
                rf'{self.re_n} *Total energy components:([\s\S]+?)((?:{self.re_n}{self.re_n}|\| Electronic free energy per atom\s*:\s*[Ee\d\.\-]+ eV))',
                repeats=False,
                str_operation=str_to_energy_components,
                convert=False,
            ),
            Quantity(
                'forces',
                rf'{self.re_n} *Total forces\([\s\d]+\)\s*:([\s\d\.\-\+Ee]+){self.re_n}',
                repeats=True,
            ),
            Quantity(
                'stress_tensor',
                rf'{self.re_n} *Sum of all contributions\s*:\s*([\d\.\-\+Ee ]+{self.re_n})',
                repeats=False,
            ),
            Quantity('pressure', r' *\|\s*Pressure:\s*([\d\.\-\+Ee ]+)', repeats=False),
            Quantity(
                'scf_convergence',
                rf'{self.re_n} *Self-consistency convergence accuracy:([\s\S]+?)(\| Change of total energy\s*:\s*[\d\.\-\+Ee V]+)',
                repeats=False,
                str_operation=str_to_scf_convergence,
                convert=False,
            ),
            Quantity(
                'humo',
                r'Highest occupied state \(VBM\) at\s*([\d\.\-\+Ee ]+) (?P<__unit>\w+)',
                repeats=False,
                dtype=float,
            ),
            Quantity(
                'lumo',
                r'Lowest unoccupied state \(CBM\) at\s*([\d\.\-\+Ee ]+) (?P<__unit>\w+)',
                repeats=False,
                dtype=float,
            ),
            Quantity(
                'fermi_level',  # older version
                rf'{self.re_n} *\| Chemical potential \(Fermi level\) in (\w+)\s*:([\d\.\-\+Ee ]+)',
                str_operation=lambda x: float(x.split()[1])
                * units_mapping.get(x.split()[0]),
            ),
            Quantity(
                'fermi_level',  # newer version
                rf'{self.re_n} *\| Chemical potential \(Fermi level\)\:\s*([\-\d\.]+)\s*(\w+)',
                str_operation=lambda x: float(x.split()[0])
                * units_mapping.get(x.split()[1], 1),
            ),
            Quantity(
                'time_calculation',
                r'Time for this iteration +: +[\d\.]+ s +([\d\.]+) s',
                dtype=float,
            ),
        ]

        calculation_quantities = [
            Quantity(
                'self_consistency',
                r'Begin self\-consistency iteration #\s*\d+([\s\S]+?Total energy evaluation[s:\d\. ]+)',
                repeats=True,
                sub_parser=TextParser(quantities=scf_quantities),
            ),
            # different format for scf loop
            Quantity(
                'self_consistency',
                rf'{self.re_n} *SCF\s*\d+\s*:([ \|\-\+Ee\d\.s]+)',
                repeats=True,
                sub_parser=TextParser(
                    quantities=[
                        Quantity(
                            'scf_convergence',
                            r'([\s\S]+)',
                            str_operation=str_to_scf_convergence2,
                            repeats=False,
                            convert=False,
                        )
                    ]
                ),
            ),
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
                'energy_components',
                rf'{self.re_n} *Total energy components:([\s\S]+?)((?:{self.re_n}{self.re_n}|\| Electronic free energy per atom\s*:\s*[\d\.\-Ee]+ eV))',
                repeats=True,
                str_operation=str_to_energy_components,
                convert=False,
            ),
            Quantity(
                'energy_xc',
                rf'{self.re_n} *Start decomposition of the XC Energy([\s\S]+?)End decomposition of the XC Energy',
                str_operation=str_to_energy_components,
                repeats=False,
                convert=False,
            ),
            eigenvalues,
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
                Program.version,
                r'(?:Version|FHI\-aims version)\s*\:*\s*([\d\.]+)\s*',
                repeats=False,
            ),
            Quantity(
                xsection_run.x_fhi_aims_program_compilation_date,
                r'Compiled on ([\d\/]+)',
                repeats=False,
            ),
            Quantity(
                xsection_run.x_fhi_aims_program_compilation_time,
                r'at (\d+\:\d+\:\d+)',
                repeats=False,
            ),
            Quantity(Program.compilation_host, r'on host ([\w\.\-]+)', repeats=False),
            date_time,
            Quantity(
                TimeRun.cpu1_start,
                r'Time zero on CPU 1\s*:\s*([0-9\-E\.]+)\s*(?P<__unit>\w+)\.',
                repeats=False,
                dtype=float,
            ),
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
