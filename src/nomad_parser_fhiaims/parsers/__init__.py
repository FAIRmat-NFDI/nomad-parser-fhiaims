from nomad.config.models.plugins import ParserEntryPoint


class EntryPoint(ParserEntryPoint):

    def load(self):
        from nomad.parsing.parser import MatchingParserInterface

        return MatchingParserInterface(
            parser_class_name='nomad_parser_fhiaims.parsers.parser.FHIAimsParser',
            **self.dict(),
        )


parser_entry_point = EntryPoint(
    name='parsers/fhiaims',
    aliases=['parsers/fhi-aims', 'parsers/fhiaims'],
    description='NOMAD parser for FHIAIMS.',
    python_package='nomad_parser_fhiaims',
    mainfile_contents_re=r'^(.*\n)*?\s*Invoking FHI-aims \.\.\.',
)
