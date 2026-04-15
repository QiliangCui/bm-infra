import time
from scripts.common.spanner_manager import SpannerManager

class KernelCasesGenerator:
    def __init__(self):
        self.db_manager = SpannerManager()

    def yield_tuning_case(self) -> list[int, str]:
        # yield a tuple representing case id and the key_value string encoding the tuing case
        raise NotImplementedError('This function should be implemented in subclass')

    def generate_cases(self):
        start_time = time.perf_counter_ns()
        valid_case_count = 0
        scan_space = 0
        self.db_manager.init_case_set(self.case_set_id, scan_space, self.case_set_desc)
        for case_id, case_key_value_str in self.yield_tuning_case():
            self.db_manager.add_tuner_case(self.case_set_id, case_id, case_key_value_str)
            valid_case_count += 1
        self.db_manager.flush()
        duration = time.perf_counter_ns() - start_time
        self.db_manager.finish_case_set(self.case_set_id, 
                                        valid_case_count, 
                                        0, # invalid case count, doesn't matter here
                                        duration / (10**9))
        print(f"\nComplete Generate Tuning Cases for {self.case_set_id}, Valid Cases: {valid_case_count} | Duration: {duration / (10**9):.2f}s")