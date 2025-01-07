from typing import TYPE_CHECKING, Any, Union

if TYPE_CHECKING:
    from nomad.datamodel.datamodel import (
        EntryArchive,
    )
    from structlog.stdlib import (
        BoundLogger,
    )

import os
import re

import numpy as np
from nomad.parsing.file_parser.mapping_parser import (
    MetainfoParser,
)
from nomad.parsing.file_parser.mapping_parser import (
    TextParser as TextMappingParser,
)
from nomad_simulations.schema_packages.general import Program

from nomad_parser_fhiaims.parsers.out_parser import RE_GW_FLAG
from nomad_parser_fhiaims.parsers.out_parser import FHIAimsOutReader

from nomad_parser_fhiaims.schema_packages.schema_package import (
    TEXT_ANNOTATION_KEY,
    TEXT_DOS_ANNOTATION_KEY,
    TEXT_GW_ANNOTATION_KEY,
    TEXT_GW_WORKFLOW_ANNOTATION_KEY,
    Simulation,
)


class FHIAimsOutConverter(TextMappingParser):
    _gw_flag_map = {
        'gw': 'G0W0',
        'gw_expt': 'G0W0',
        'ev_scgw0': 'ev-scGW',
        'ev_scgw': 'ev-scGW',
        'scgw': 'scGW',
    }

    _xc_map = {
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

    _section_names = ['full_scf', 'geometry_optimization', 'molecular_dynamics']

    def get_fhiaims_file(self, default: str) -> list[str]:
        maindir = os.path.dirname(self.filepath)
        base, *ext = default.split('.')
        ext = '.'.join(ext)
        base = base.lower()
        files = os.listdir(maindir)
        files = [os.path.basename(f) for f in files]
        files = [
            os.path.join(maindir, f)
            for f in files
            if base.lower() in f.lower() and f.endswith(ext)
        ]
        files.sort()
        return files

    def get_xc_functionals(self, xc: str) -> list[dict[str, Any]]:
        return [
            dict(name=functional.get('name')) for functional in self._xc_map.get(xc, [])
        ]

    def get_dos(
        self,
        total_dos_files: list[list[str]],
        atom_dos_files: list[list[str]],
        species_dos_files: list[list[str]],
    ) -> list[dict[str, Any]]:
        def load_dos(dos_file: str) -> list[dict[str, Any]]:
            dos_files = self.get_fhiaims_file(dos_file)
            if not dos_files:
                return []
            try:
                data = np.loadtxt(dos_files[0]).T
            except Exception:
                return []
            if not np.size(data):
                return []
            return [
                dict(energies=data[0], values=value, nenergies=len(data[0]))
                for value in data[1:]
            ]

        def get_pdos(dos_files: list[str], dos_labels: list[str], type=str):
            dos = []
            for dos_file in dos_files:
                labels = [label for label in dos_labels if label in dos_file]
                pdos = load_dos(dos_file)
                if not pdos:
                    continue
                for n, data in enumerate(pdos):
                    # TODO use these to link pdos to system
                    data['type'] = type
                    data['label'] = labels[n % len(labels)]
                    data['spin'] = 1 if 'spin_dn' in dos_file else 0
                    data['orbital'] = n - 1 if n else None
                dos.extend(pdos)
            return dos

        projected_dos = []
        # atom-projected dos
        if atom_dos_files:
            projected_dos.extend(get_pdos(*atom_dos_files, type='atom'))

        # species-projected dos
        if species_dos_files:
            projected_dos.extend(get_pdos(*species_dos_files, type='species'))

        # total dos
        total_dos = []
        for dos_file in (
            total_dos_files[0] if total_dos_files else ['KS_DOS_total_raw.dat']
        ):
            dos = load_dos(dos_file)
            for n, data in enumerate(dos):
                data['spin'] = n
                pdata = data.setdefault('projected_dos', [])
                pdata.extend([d for d in projected_dos if d['spin'] == data['spin']])
            total_dos.extend(dos)

        return total_dos

    def get_eigenvalues(
        self, source: list[dict[str, Any]], params: dict[str, Any]
    ) -> list[dict[str, Any]]:
        n_spin = params.get('Number of spin channels', 1)
        eigenvalues = []
        for data in source:
            kpts = data.get('kpoints', [np.zeros(3)] * n_spin)
            kpts = np.reshape(kpts, (len(kpts) // n_spin, n_spin, 3))
            kpts = np.transpose(kpts, axes=(1, 0, 2))[0]

            occs_eigs = data.get('occupation_eigenvalue')
            n_kpts = len(kpts)
            n_eigs = len(occs_eigs) // (n_kpts * n_spin)
            occs_eigs = np.transpose(
                np.reshape(occs_eigs, (n_kpts, n_spin, n_eigs, 2)), axes=(3, 1, 0, 2)
            )
            for spin in range(n_spin):
                eigenvalues.append(
                    dict(
                        nbands=n_eigs,
                        npoints=n_kpts,
                        points=kpts,
                        occupations=occs_eigs[0][spin],
                        eigenvalues=occs_eigs[1][spin],
                    )
                )
        return eigenvalues

    def get_energies(self, source: dict[str, Any]) -> dict[str, Any]:
        total_keys = ['Total energy uncorrected', 'Total energy']
        energies = {}
        components = []
        for key, val in source.get('energy', {}).items():
            if key in total_keys:
                energies.setdefault('value', val)
            else:
                components.append({'name': key, 'value': val})
        for key, val in source.get('energy_components', [{}])[-1].items():
            components.append({'name': key, 'value': val})
        energies['components'] = components
        return energies

    def get_gw_flag(self, gw_flag: str):
        return self._gw_flag_map.get(gw_flag)

    def get_sections(self, source: dict[str, Any], **kwargs) -> list[dict[str, Any]]:
        result = []
        include = kwargs.get('include')
        for name in self._section_names:
            for data in source.get(name, []):
                res = {}
                for key in data.keys():
                    if include and key not in include:
                        continue
                    val = data.get(key, self.data.get(key))
                    if val is not None:
                        res[key] = val
                if res:
                    result.append(res)
        return result


class FHIAimsParser:
    def get_mainfile_keys(self, **kwargs) -> Union[bool, list[str]]:
        buffer = kwargs.get('decoded_buffer', '')
        match = re.search(RE_GW_FLAG, buffer)
        if match:
            gw_flag = match[1]
        else:
            overlap = len(RE_GW_FLAG) + 1
            block = max(len(buffer), 4916)
            match = None
            position = len(buffer)
            with open(kwargs.get('filename')) as f:
                while True:
                    f.seek(position - overlap)
                    text = f.read(block + overlap)
                    match = re.search(RE_GW_FLAG, text)
                    position += block
                    if not text or match:
                        gw_flag = match[1]
                        break
        if gw_flag in RE_GW_FLAG.keys():
            return ['GW', 'GW_workflow']
        return True

    def parse(
        self,
        mainfile: str,
        archive: 'EntryArchive',
        logger: 'BoundLogger',
        child_archives: dict[str, 'EntryArchive'] = {},
        **kwargs,
    ) -> None:
        out_converter = FHIAimsOutConverter()
        out_converter.text_parser = FHIAimsOutReader()
        out_converter.filepath = mainfile
        self.out_parser = out_converter

        archive_data_handler = MetainfoParser()
        archive_data_handler.annotation_key = kwargs.get(
            'annotation_key', TEXT_ANNOTATION_KEY
        )
        archive_data_handler.data_object = Simulation(program=Program(name='FHI-aims'))

        out_converter.convert(archive_data_handler, remove=True)

        archive.data = archive_data_handler.data_object
        self.out_parser = out_converter

        # separate parsing of dos due to a problem with mapping physical
        # property variables
        archive_data_handler.annotation_key = TEXT_DOS_ANNOTATION_KEY
        out_converter.convert(archive_data_handler, remove=True)

        gw_archive = child_archives.get('GW')
        if gw_archive is not None:
            # GW single point
            parser = FHIAimsParser()
            parser.parse(
                mainfile, gw_archive, logger, annotation_key=TEXT_GW_ANNOTATION_KEY
            )

            # DFT-GW workflow
            gw_workflow_archive = self._child_archives.get('GW_workflow')
            parser = FHIAimsParser()
            parser.parse(
                mainfile,
                gw_workflow_archive,
                logger,
                annotation_key=TEXT_GW_WORKFLOW_ANNOTATION_KEY,
            )

        # TODO remove this only for debug
        self.out_parser = out_converter
        self.archive_data_parser = archive_data_handler
        # close file contexts
        # out_converter.close()
        # archive_data_handler.close()
