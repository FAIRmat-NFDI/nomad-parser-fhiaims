from nomad.config.models.plugins import SchemaPackageEntryPoint
from pydantic import Field


class MySchemaPackageEntryPoint(SchemaPackageEntryPoint):
    parameter: int = Field(0, description='Custom configuration parameter')

    def load(self):
        from nomad_parser_fhiaims.schema_packages.schema_package import m_package
 
        return m_package


schema_package_entry_point = MySchemaPackageEntryPoint(
    name='FHIAims schema package',
    description='Schema package defined using the new plugin mechanism.',
)
