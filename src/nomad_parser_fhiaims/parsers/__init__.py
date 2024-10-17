from nomad.config.models.plugins import ParserEntryPoint
from pydantic import Field



class EntryPoint(ParserEntryPoint):
    parameter: int = Field(0, description='Custom configuration parameter')
    level: int = Field(
        0,
        description="""
        Order of execution of parser with respect to other parsers.
    """,
    )

    def load(self):
        from nomad.parsing.parser import MatchingParserInterface

        return MatchingParserInterface(
            parser_class_name='nomad_parser_fhiaims.parsers.parser.FHIAimsParser',
            **self.dict(),
        )


parser_entry_point = EntryPoint(
    name='parsers/fhi-aims',
    aliases=['parsers/fhi-aims'],
    description='NOMAD parser for FHIAIMS.',
    python_package='nomad_parser_fhiaims',
    mainfile_contents_re=r'^(.*\n)*?\s*Invoking FHI-aims \.\.\.',
)
