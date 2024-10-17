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
from nomad.metainfo import SchemaPackage
from nomad_simulations.schema_packages import general, model_method, model_system, outputs, properties
from nomad.parsing.file_parser.mapping_parser import MappingAnnotationModel

configuration = config.get_plugin_entry_point(
    'nomad_parser_fhiaims.schema_packages:schema_package_entry_point'
)

m_package = SchemaPackage()

TEXT_ANNOTATION_KEY = 'text'

class Program(general.Program):

    general.Program.version.m_annotations[TEXT_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.version')


class XCFunctional(model_method.XCFunctional):

    model_method.XCFunctional.libxc_name.m_annotations[TEXT_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.name')

class DFT(model_method.DFT):

    model_method.DFT.xc_functionals.m_annotations[TEXT_ANNOTATION_KEY] = MappingAnnotationModel(mapper=('get_xc_functionals', ['.controlInOut_xc']))


class AtomicCell(model_system.AtomicCell):

    model_system.AtomicCell.lattice_vectors.m_annotations[TEXT_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.lattice_vectors')

    model_system.AtomicCell.positions.m_annotations[TEXT_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.structure.positions', unit='angstrom')


class ModelSystem(model_system.ModelSystem):

    model_system.AtomicCell.m_def.m_annotations[TEXT_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.@')


class EnergyContribution(properties.energies.EnergyContribution):

    properties.energies.EnergyContribution.name.m_annotations[TEXT_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.name')


class TotalEnergy(properties.energies.TotalEnergy):

    properties.energies.TotalEnergy.value.m_annotations[TEXT_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.value')

    properties.energies.TotalEnergy.contributions.m_annotations[TEXT_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.contributions')


class Outputs(outputs.Outputs):

    outputs.Outputs.total_energies.m_annotations[TEXT_ANNOTATION_KEY] = MappingAnnotationModel(mapper=('get_energies', ['.@']))


class Simulation(general.Simulation):
    general.Simulation.program.m_annotations[TEXT_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.@')

    model_method.DFT.m_def.m_annotations[TEXT_ANNOTATION_KEY] = MappingAnnotationModel(mapper='.@')

    general.Simulation.model_system.m_annotations[TEXT_ANNOTATION_KEY] = MappingAnnotationModel(mapper=('get_sections', ['.@'], dict(include=['lattice_vectors', 'structure'])))

    general.Simulation.outputs.m_annotations[TEXT_ANNOTATION_KEY] = MappingAnnotationModel(mapper=('get_sections', ['.@'], dict(include=['energy', 'energy_components'])))


Simulation.m_def.m_annotations[TEXT_ANNOTATION_KEY] = MappingAnnotationModel(mapper='@')


m_package.__init_metainfo__()
