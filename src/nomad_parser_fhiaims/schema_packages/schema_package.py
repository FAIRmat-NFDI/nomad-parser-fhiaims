from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    pass

from nomad.config import config
from nomad.metainfo import SchemaPackage, Property
from nomad.datamodel.metainfo.annotations import Mapper, AnnotationModel
from nomad_simulations.schema_packages import (
    general,
    model_method,
    model_system,
    outputs,
    properties,
    variables,
)
from nomad.parsing.file_parser.mapping_parser import MAPPING_ANNOTATION_KEY

configuration = config.get_plugin_entry_point(
    'nomad_parser_fhiaims.schema_packages:schema_package_entry_point'
)

m_package = SchemaPackage()

TEXT_ANNOTATION_KEY = 'text'
TEXT_DOS_ANNOTATION_KEY = 'text_dos'
TEXT_GW_ANNOTATION_KEY = 'text_gw'
TEXT_GW_WORKFLOW_ANNOTATION_KEY = 'text_gw_workflow'


class Program(general.Program):
    general.Program.version.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {})[
        TEXT_ANNOTATION_KEY
    ] = Mapper(mapper='.version')


class XCFunctional(model_method.XCFunctional):
    model_method.XCFunctional.libxc_name.m_annotations.setdefault(
        MAPPING_ANNOTATION_KEY, {}
    )[TEXT_ANNOTATION_KEY] = Mapper(mapper='.name')


class DFT(model_method.DFT):
    model_method.DFT.xc_functionals.m_annotations.setdefault(
        MAPPING_ANNOTATION_KEY, {}
    )[TEXT_ANNOTATION_KEY] = Mapper(mapper=('get_xc_functionals', ['.controlInOut_xc']))


class GW(model_method.GW):
    model_method.GW.type.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {})[
        TEXT_GW_ANNOTATION_KEY
    ] = Mapper(mapper=('get_gw_flag', ['.gw_flag']))


class AtomicCell(model_system.AtomicCell):
    model_system.AtomicCell.lattice_vectors.m_annotations.setdefault(
        MAPPING_ANNOTATION_KEY, {}
    )[TEXT_ANNOTATION_KEY] = Mapper(mapper='.lattice_vectors')

    model_system.AtomicCell.positions.m_annotations.setdefault(
        MAPPING_ANNOTATION_KEY, {}
    )[TEXT_ANNOTATION_KEY] = Mapper(mapper='.structure.positions', unit='angstrom')


class ModelSystem(model_system.ModelSystem):
    model_system.AtomicCell.m_def.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {})[
        TEXT_ANNOTATION_KEY
    ] = Mapper(mapper='.@')


class EnergyContribution(properties.energies.EnergyContribution):
    properties.energies.EnergyContribution.name.m_annotations.setdefault(
        MAPPING_ANNOTATION_KEY, {}
    )[TEXT_ANNOTATION_KEY] = Mapper(mapper='.name')


class TotalEnergy(properties.energies.TotalEnergy):
    properties.energies.TotalEnergy.value.m_annotations.setdefault(
        MAPPING_ANNOTATION_KEY, {}
    )[TEXT_ANNOTATION_KEY] = Mapper(mapper='.value')

    properties.energies.TotalEnergy.contributions.m_annotations.setdefault(
        MAPPING_ANNOTATION_KEY, {}
    )[TEXT_ANNOTATION_KEY] = Mapper(mapper='.components')


class TotalForce(properties.forces.TotalForce):
    properties.forces.TotalForce.rank.m_annotations.setdefault(
        MAPPING_ANNOTATION_KEY, {}
    )[TEXT_ANNOTATION_KEY] = Mapper(mapper='.rank')
    properties.forces.TotalForce.value.m_annotations.setdefault(
        MAPPING_ANNOTATION_KEY, {}
    )[TEXT_ANNOTATION_KEY] = Mapper(mapper='.forces')


class Energy2(variables.Energy2):
    variables.Energy2.n_points.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {})[
        TEXT_DOS_ANNOTATION_KEY
    ] = Mapper(mapper='.nenergies')
    variables.Energy2.points.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {})[
        TEXT_DOS_ANNOTATION_KEY
    ] = Mapper(mapper='.energies')


class DosProfile(properties.spectral_profile.DOSProfile):
    properties.spectral_profile.DOSProfile.value.m_annotations.setdefault(
        MAPPING_ANNOTATION_KEY, {}
    )[TEXT_DOS_ANNOTATION_KEY] = Mapper(mapper='.values')


class ElectronicDensityOfStates(properties.spectral_profile.ElectronicDensityOfStates):
    properties.spectral_profile.ElectronicDensityOfStates.value.m_annotations.setdefault(
        MAPPING_ANNOTATION_KEY, {}
    )[TEXT_DOS_ANNOTATION_KEY] = Mapper(mapper='.values')

    variables.Energy2.m_def.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {})[
        TEXT_DOS_ANNOTATION_KEY
    ] = Mapper(mapper='.@')

    properties.spectral_profile.ElectronicDensityOfStates.projected_dos.m_annotations.setdefault(
        MAPPING_ANNOTATION_KEY, {}
    )[TEXT_DOS_ANNOTATION_KEY] = Mapper(mapper='.projected_dos')


class Variables(variables.Variables):
    variables.Variables.n_points.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {})[
        TEXT_ANNOTATION_KEY
    ] = Mapper(mapper='.npoints')
    # TODO this does not work as points is scalar
    # variables.Variables.points.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {})[TEXT_ANNOTATION_KEY] = (
    #   Mapper(mapper='.points')
    # )


class ElectronicEigenvalues(properties.ElectronicEigenvalues):
    properties.ElectronicEigenvalues.n_bands.m_annotations.setdefault(
        MAPPING_ANNOTATION_KEY, {}
    )[TEXT_ANNOTATION_KEY] = Mapper(mapper='.nbands')
    properties.ElectronicEigenvalues.variables.m_annotations.setdefault(
        MAPPING_ANNOTATION_KEY, {}
    )[TEXT_ANNOTATION_KEY] = Mapper(mapper='.@', search='.@[0]')
    properties.ElectronicEigenvalues.value.m_annotations.setdefault(
        MAPPING_ANNOTATION_KEY, {}
    )[TEXT_ANNOTATION_KEY] = Mapper(mapper='.eigenvalues')
    properties.ElectronicEigenvalues.occupation.m_annotations.setdefault(
        MAPPING_ANNOTATION_KEY, {}
    )[TEXT_ANNOTATION_KEY] = Mapper(mapper='.occupations')


class Outputs(outputs.Outputs):
    outputs.Outputs.total_energies.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {})[
        TEXT_ANNOTATION_KEY
    ] = Mapper(mapper=('get_energies', ['.@']))

    outputs.Outputs.total_forces.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {})[
        TEXT_ANNOTATION_KEY
    ] = Mapper(mapper=('get_forces', ['.@']))

    outputs.Outputs.electronic_dos.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {})[
        TEXT_DOS_ANNOTATION_KEY
    ] = Mapper(
        mapper=(
            'get_dos',
            [
                '.total_dos_files',
                '.atom_projected_dos_files',
                '.species_projected_dos_files',
            ],
        )
    )

    outputs.Outputs.electronic_eigenvalues.m_annotations.setdefault(
        MAPPING_ANNOTATION_KEY, {}
    )[TEXT_ANNOTATION_KEY] = Mapper(
        mapper=('get_eigenvalues', ['.eigenvalues', 'array_size_parameters'])
    )


class Simulation(general.Simulation):
    general.Simulation.program.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {})[
        TEXT_ANNOTATION_KEY
    ] = Mapper(mapper='.@')

    model_method.DFT.m_def.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {})[
        TEXT_ANNOTATION_KEY
    ] = Mapper(mapper='.@')

    model_method.GW.m_def.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {})[
        TEXT_GW_ANNOTATION_KEY
    ] = Mapper(mapper='.@')

    general.Simulation.model_system.m_annotations.setdefault(
        MAPPING_ANNOTATION_KEY, {}
    )[TEXT_ANNOTATION_KEY] = Mapper(
        mapper=(
            'get_sections',
            ['.@'],
            dict(include=['lattice_vectors', 'structure']),
        )
    )

    general.Simulation.outputs.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {})[
        TEXT_ANNOTATION_KEY
    ] = Mapper(
        mapper=(
            'get_sections',
            ['.@'],
            dict(
                include=[
                    'energy',
                    'energy_components',
                    'forces',
                    'eigenvalues',
                ]
            ),
        )
    )

    general.Simulation.outputs.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {})[
        TEXT_DOS_ANNOTATION_KEY
    ] = Mapper(
        mapper=(
            'get_sections',
            ['.@'],
            dict(
                include=[
                    'total_dos_files',
                    'species_projected_dos_files',
                ]
            ),
        )
    )


Simulation.m_def.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {})[
    TEXT_ANNOTATION_KEY
] = Mapper(mapper='@')

Simulation.m_def.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {})[
    TEXT_DOS_ANNOTATION_KEY
] = Mapper(mapper='@')

Simulation.m_def.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {})[
    TEXT_GW_ANNOTATION_KEY
] = Mapper(mapper='@')

Simulation.m_def.m_annotations.setdefault(MAPPING_ANNOTATION_KEY, {})[
    TEXT_GW_WORKFLOW_ANNOTATION_KEY
] = Mapper(mapper='@')


m_package.__init_metainfo__()
