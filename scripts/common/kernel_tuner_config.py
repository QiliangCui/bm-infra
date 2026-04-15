from dataclasses import dataclass, fields

@dataclass
class Key:
    # Specify the key for tuning case
    pass

@dataclass
class Tilings:
    # Specify the tiles for tuning case
    pass

class Case:
    intra_delimiter = ',' # delimiter between key/tilings fields
    inter_delimiter = ':' # delimiter between key and tilings

    @classmethod
    def to_string(cls, key, tilings):
        key_field_values = list(getattr(key, field_name) for field_name in map(lambda x: x.name, fields(key)))
        tilings_field_values = list(getattr(tilings, field_name) for field_name in map(lambda x: x.name, fields(tilings)))
        def mapper(x):
            return str(x) if not isinstance(x, str) else f'\'{x}\''
        return cls.inter_delimiter.join([cls.intra_delimiter.join(map(mapper, key_field_values)), 
                                        cls.intra_delimiter.join(map(mapper, tilings_field_values))])
    
    @classmethod
    def from_string(cls, string, key_class, tilings_class):
        key_field_values_string, tilings_field_values_string = string.split(cls.inter_delimiter)

        key_field_values = key_field_values_string.split(cls.intra_delimiter)
        key_fields_map = {}
        for key_field, key_field_value in zip(fields(key_class), key_field_values):
            key_field_value = key_field_value.strip('\'').strip('\"').strip('\'')
            key_fields_map[key_field.name] = key_field.type(key_field_value) if key_field_value != 'None' else None
        key = key_class(**key_fields_map)

        tilings_field_values = tilings_field_values_string.split(cls.intra_delimiter)
        tilings_fields_map = {}
        for tilings_field, tilings_field_value in zip(fields(tilings_class), tilings_field_values):
            tilings_fields_map[tilings_field.name] = tilings_field.type(tilings_field_value) if tilings_field_value != 'None' else None
        tilings = tilings_class(**tilings_fields_map)

        return key, tilings

class KernelTunerConfig:
    def setup_inputs(self, case):
        raise NotImplementedError("Specific kernel should implement this to initialize the inputs to kernel based on the case's key")

    def get_runner(self, iters: int) -> [int, int]: # type: ignore
        # return an integer represent the average kernel run time and the total time
        raise NotImplementedError("Specific kernel should implement this to call the kernl with the inputs from setup_inputs")

    def caseset_id(self):
        raise NotImplementedError("Specific kernel should implement this to return the caseset_id")

    def run_id(self):
        raise NotImplementedError("Specific kernel should implement this to return the run_id")