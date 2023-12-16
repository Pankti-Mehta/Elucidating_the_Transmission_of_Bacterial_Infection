# This code is part of Rice COMP182 and is made available for your
# use as a student in COMP182. You are specifically forbidden from
# posting this code online in a public fashion (e.g., on a public
# GitHub repository) or otherwise making it, or any derivative of it,
# available to future COMP182 students.

import traceback
import re
import sys
from collections import defaultdict

print("""DISCLAIMER: This tool is intended to ensure your code is
compatible with the autograder, not that it is correct. It is
possible to 'pass' this tool, yet receive a 0 on the coding portion.
You are still responsible for writing your own tests to ensure the
correctness of your code.
""")

class SkeletonAutograder():
    def __init__(self):
        self._allowed_imports = []
        self._test_cases_functions = []
        self._test_cases_inputs = []
        self._test_cases_expected = []
        self._test_cases_expected_alternate = []
        self._test_cases_notes = []

    def set_allowed_imports(self, imports: list):
        self._allowed_imports = imports

    def add_test_case(self, function, inputs: list, outputs: list, note = "", alternate_solutions = []):
        self._test_cases_functions.append(function)
        self._test_cases_inputs.append(inputs)
        self._test_cases_expected.append(outputs)
        self._test_cases_expected_alternate.append(alternate_solutions)
        self._test_cases_notes.append(note)

    def fail_test(self):
        print("\nFAILED!")
        exit()

    def check_python_version(self):
        """
        Checks the Python version.
        """
        if sys.version_info[0] < 3:
            print("You must run this script using Python 3.")
            exit()
        
        if sys.version_info[1] < 8:
            print("""\nWARNING: Your solution will be graded on Python 3.8, and your version is """ + str(sys.version_info[0])+"."+str(sys.version_info[1])+""". 
            We encourage you to test your code on Python 3.8+ for consistency.\n""")

    def check_imports(self):
        """
        This method verifies that only allowed imports are used in 
        student's submission. Requires 'set_allowed_imports' to be 
        executed before checking.

        If there is an illegal import, then it fails and exits the 
        skeleton autograder.
        """

        # Set regular expression to match Python import statements.
        pattern = re.compile(r"(^from\s(\w+)\simport\s([\w*]+)$)|(^import\s(\w+)$)|^import\s(\w+)\sas\s([\w*]+)$")

        # Define list for illegally used imports.
        illegal_imports = []

        with open("autograder.py") as f:
            lines = f.readlines()
            for line in lines:
                # Match the pattern.
                line = re.sub(r'\s+$', '', re.sub(r'^\s+', '', line))
                match = pattern.match(line)

                # Check for matches.
                if match is not None:
                    groups = match.groups(default='')
                    importstr = " ".join(groups[1:3] if groups[0] else [groups[4]])
                    if importstr not in self._allowed_imports:
                        illegal_imports.append(line)

        if len(illegal_imports) > 0:
            print("A disallowed import was detected. Please remove this import and re-run the autograder.\nThe line(s) in question are:")
            for line in illegal_imports:
                print(line)

            self.fail_test()

    def check_directory(self):
        """
        This method verifies that student submission is in the same directory as the skeleton autograder.

        If the skeleton autograder cannot import 'autograder.py', then it fails and exists the skeleton autograder.
        """
        try:
            import autograder
        except ImportError:
            print("""Failed to import 'autograder.py'.
            Ensure the following:
                1. Your submission is titled 'autograder'.py
                2. Your submitted 'autograder.py' file is in the same directory as this file ('skeleton_autograder.py')
                3. Your submission doesn't import anything other than the imports in the original provided template file
            See the error below for more information:\n"""+traceback.format_exc())

            self.fail_test()

        except Exception:
            print("""Failed to import 'autograder.py'.
            Your code likely failed due to code located outside a function failing.
            Ensure the following:
                1. All of your code is in one of the autograder or helper functions
                2. Any testing code, or code outside of a function, is commented out
            See the error below for more information:\n"""+traceback.format_exc())

            self.fail_test()

    def run_tests(self, run_typechecks = False):
        """
        This method runs all the test cases defined. By default, it checks whether the autograder.py is located in
        the same directory and whether imports are legal.

        Flag run_typechecks can be toggled to
        """

        # Run default tests to ensure form.
        self.check_directory()
        self.check_imports()

        import autograder

        for test_id, func_name in enumerate(self._test_cases_functions):

            # Try to get the function, if the function cannot be located in autograder, then fail the test.
            try:
                func = getattr(autograder, func_name)
            except AttributeError:
                print("Could not locate function '" + func_name + "', ensure your code contains a function with that exact name.")
                print("See the error below for more information:\n")
                print(traceback.format_exc())

                self.fail_test()

            inputs = self._test_cases_inputs[test_id]
            expected = self._test_cases_expected[test_id]
            alternate = self._test_cases_expected_alternate[test_id]
            notes = self._test_cases_notes[test_id]

            # Run student's function.
            print("Running Test #"+str(test_id)+" on '"+ func_name + "'...")
            try:
                actual = func(*inputs)

                print("Input(s): "+ str(inputs))
                print("Expected Output(s): "+ str(expected))
                print("Actual Output(s)  : "+ str(actual))
                if notes != "":
                    print("** Note: "+ notes)
                print("")

                if run_typechecks and type(expected) is not type(actual):
                    print("Wrong type returned, expecting '" + str(type(expected)) + "', received '" + str(type(actual)) + "'.")

                    self.fail_test()

                if type(expected) == list or type(expected) == tuple:
                    if len(expected) != len(actual):
                        print("Was expecting "+str(len(expected))+" number of output, received "+str(len(actual))+".")

                        self.fail_test()

                if expected != actual:
                    if not alternate:
                        print("Wrong value returned, expecting '" + str(expected) + "', received '" + str(actual) + "'.")
                        self.fail_test()
                    else:
                        valid_case = False

                        for acase in range(len(alternate)):
                            if valid_case:
                                break

                            if alternate[acase] == actual:
                                valid_case = True

                        if not valid_case:
                            print("Wrong value returned, expecting '" + str(expected) + "', received '" + str(actual) + "'.")
                            self.fail_test()

                print("Test passed!\n")

            except Exception:
                print("Code failed to run, see the error below for more information:\n")
                print(traceback.format_exc())

                self.fail_test()



skeleton_autograder = SkeletonAutograder()
skeleton_autograder.set_allowed_imports(['collections *', 'copy *', 'typing Tuple'])

# Test graphs
g0 = {0:{1:10}, 1:{}}
g1 = {0:{1:0}, 1:{2:0}, 2:{1:0}}
g2 = {0:{}, 1:{2:10}, 2:{3:10}, 3:{1:10}}
g3 = {0:{1:0}, 1:{2:0}, 2:{2:0, 1:0}}

cycle2 = [1, 2, 3]
cycle3 = [1, 2]

contracted3 = {0:{3:0}, 3:{}}
cstar3 = 3

## COMP 182 Spring 2021 - Homework 5, Problem 3 Test Cases
skeleton_autograder.add_test_case('reverse_digraph_representation', [g0], {0: {}, 1: {0: 10}})
skeleton_autograder.add_test_case('modify_edge_weights', [g0, 0], None)
skeleton_autograder.add_test_case('compute_rdst_candidate', [g1, 0], {0: {}, 1: {2: 0}, 2: {1: 0}})
skeleton_autograder.add_test_case('compute_cycle', [g2], (1, 2, 3), note = "Your cycle simply needs to be in the right direction (in other words, it should be any proper rotation of the expected output).", alternate_solutions = [(2,3,1), (3,1,2)])
skeleton_autograder.add_test_case('contract_cycle', [g2, cycle2], ({0: {}, 4: {}}, 4))
skeleton_autograder.add_test_case('expand_graph', [g3, contracted3, cycle3, cstar3], {0: {1: 0}, 1: {2: 0}, 2: {}})

skeleton_autograder.run_tests(run_typechecks = False)