from typing import (
    TYPE_CHECKING, List, Dict, Any
)

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import (
        EntryArchive,
    )
    from structlog.stdlib import (
        BoundLogger,
    )

from nomad.config import config
from nomad.parsing.file_parser.mapping_parser import TextParser as TextMappingParser, MetainfoParser
from nomad_parser_fhiaims.parsers.out_parser import FHIAimsOutParser as FHIAimsOutTextParser
from nomad_parser_fhiaims.schema_packages.schema_package import Simulation

from nomad_simulations.schema_packages.general import Program

configuration = config.get_plugin_entry_point(
    'nomad_parser_fhiaims.parsers:parser_entry_point'
)



class FHIAimsOutParser(TextMappingParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._xc_map = {
            'Perdew-Wang parametrisation of Ceperley-Alder LDA': [
                {'name': 'LDA_C_PW'},
                {'name': 'LDA_X'},
            ],
            'Perdew-Zunger parametrisation of Ceperley-Alder LDA': [
                {'name': 'LDA_C_PZ'},
                {'name': 'LDA_X'},
            ],
            'VWN-LDA parametrisation of VWN5 form': [
                {'name': 'LDA_C_VWN'},
                {'name': 'LDA_X'},
            ],
            'VWN-LDA parametrisation of VWN-RPA form': [
                {'name': 'LDA_C_VWN_RPA'},
                {'name': 'LDA_X'},
            ],
            'AM05 gradient-corrected functionals': [
                {'name': 'GGA_C_AM05'},
                {'name': 'GGA_X_AM05'},
            ],
            'BLYP functional': [{'name': 'GGA_C_LYP'}, {'name': 'GGA_X_B88'}],
            'PBE gradient-corrected functionals': [
                {'name': 'GGA_C_PBE'},
                {'name': 'GGA_X_PBE'},
            ],
            'PBEint gradient-corrected functional': [
                {'name': 'GGA_C_PBEINT'},
                {'name': 'GGA_X_PBEINT'},
            ],
            'PBEsol gradient-corrected functionals': [
                {'name': 'GGA_C_PBE_SOL'},
                {'name': 'GGA_X_PBE_SOL'},
            ],
            'RPBE gradient-corrected functionals': [
                {'name': 'GGA_C_PBE'},
                {'name': 'GGA_X_RPBE'},
            ],
            'revPBE gradient-corrected functionals': [
                {'name': 'GGA_C_PBE'},
                {'name': 'GGA_X_PBE_R'},
            ],
            'PW91 gradient-corrected functionals': [
                {'name': 'GGA_C_PW91'},
                {'name': 'GGA_X_PW91'},
            ],
            'M06-L gradient-corrected functionals': [
                {'name': 'MGGA_C_M06_L'},
                {'name': 'MGGA_X_M06_L'},
            ],
            'M11-L gradient-corrected functionals': [
                {'name': 'MGGA_C_M11_L'},
                {'name': 'MGGA_X_M11_L'},
            ],
            'TPSS gradient-corrected functionals': [
                {'name': 'MGGA_C_TPSS'},
                {'name': 'MGGA_X_TPSS'},
            ],
            'TPSSloc gradient-corrected functionals': [
                {'name': 'MGGA_C_TPSSLOC'},
                {'name': 'MGGA_X_TPSS'},
            ],
            'hybrid B3LYP functional': [{'name': 'HYB_GGA_XC_B3LYP5'}],
            'Hartree-Fock': [{'name': 'HF_X'}],
            'HSE': [{'name': 'HYB_GGA_XC_HSE03'}],
            'HSE-functional': [{'name': 'HYB_GGA_XC_HSE06'}],
            'hybrid-PBE0 functionals': [
                {'name': 'GGA_C_PBE'},
                {
                    'name': 'GGA_X_PBE',
                    'weight': lambda x: 0.75 if x is None else 1.0 - x,
                },
                {'name': 'HF_X', 'weight': lambda x: 0.25 if x is None else x},
            ],
            'hybrid-PBEsol0 functionals': [
                {'name': 'GGA_C_PBE_SOL'},
                {
                    'name': 'GGA_X_PBE_SOL',
                    'weight': lambda x: 0.75 if x is None else 1.0 - x,
                },
                {'name': 'HF_X', 'weight': lambda x: 0.25 if x is None else x},
            ],
            'Hybrid M06 gradient-corrected functionals': [
                {'name': 'MGGA_C_M06'},
                {'name': 'HYB_MGGA_X_M06'},
            ],
            'Hybrid M06-2X gradient-corrected functionals': [
                {'name': 'MGGA_C_M06_2X'},
                {'name': 'HYB_MGGA_X_M06'},
            ],
            'Hybrid M06-HF gradient-corrected functionals': [
                {'name': 'MGGA_C_M06_HF'},
                {'name': 'HYB_MGGA_X_M06'},
            ],
            'Hybrid M08-HX gradient-corrected functionals': [
                {'name': 'MGGA_C_M08_HX'},
                {'name': 'HYB_MGGA_X_M08_HX'},
            ],
            'Hybrid M08-SO gradient-corrected functionals': [
                {'name': 'MGGA_C_M08_SO'},
                {'name': 'HYB_MGGA_X_M08_SO'},
            ],
            'Hybrid M11 gradient-corrected functionals': [
                {'name': 'MGGA_C_M11'},
                {'name': 'HYB_MGGA_X_M11'},
            ],
        }

    def get_xc_functionals(self, xc: str) -> List[Dict[str, Any]]:
        return [dict(name=functional.get('name')) for functional in self._xc_map.get(xc, [])]

    def get_energies(self, source: Dict[str, Any]) -> Dict[str, Any]:
        total_keys = ['Total energy uncorrected', 'Total energy']
        energies = {}
        contributions = []
        for key, val in source.get('energy', {}).items():
            if key in total_keys:
                energies.setdefault('value', val)
            else:
                contributions.append({'name': key, 'value': val})
        for key, val in source.get('energy_components', [{}])[-1].items():
            contributions.append({'name': key, 'value': val})
        energies['contributions'] = contributions
        return energies

    def get_sections(self, source: Dict[str, Any], **kwargs) -> List[Dict[str, Any]]:
        result = []
        section_names = ['full_scf', 'geometry_optimization', 'molecular_dynamics']
        for name in section_names:
            for data in source.get(name, []):
                res = {}
                for key in kwargs.get('include', []):
                    val = data.get(key, self.data.get(key))
                    if val is not None:
                        res[key] = val
                if res:
                    result.append(res)
        return result


class FHIAimsParser:
    def parse(
        self,
        mainfile: str,
        archive: 'EntryArchive',
        logger: 'BoundLogger',
        child_archives: dict[str, 'EntryArchive'] = None,
    ) -> None:
        logger.info('NewParser.parse', parameter=configuration.parameter)

        out_parser = FHIAimsOutParser()
        out_parser.text_parser = FHIAimsOutTextParser()
        out_parser.filepath = mainfile

        archive_data_parser = MetainfoParser()
        archive_data_parser.annotation_key = 'text'
        archive_data_parser.data_object = Simulation(program=Program(name='FHI-aims'))

        out_parser.convert(archive_data_parser)

        archive.data = archive_data_parser.data_object

        self.out_parser = out_parser
        self.archive_data_parser = archive_data_parser

        out_parser.close()
        archive_data_parser.close()
