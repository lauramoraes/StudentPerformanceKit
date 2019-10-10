import unittest
import os
import numpy as np
from spkit import bkt
import logging

LOGGER = logging.getLogger(__name__)


class TestBKT(unittest.TestCase):
    """ Unit test class to test BKT module """
    def setUp(self):
        self.PARAMS_KEYS = {
            'skills': list, 'SolverId': str, 'nK': str, 'nG': str,
            'nS': str, 'nO': str, 'nZ': str, 'Null skill ratios': str
        }
        self.SKILLS_KEYS = {
            "skill": str, "priors": np.ndarray,
            "transitions": np.ndarray, "emissions": np.ndarray
        }

    def test_download(self):
        """ Testing HMM-scalable download """
        model = bkt.BKT()
        model.download()

        # Check if directory exists and it contains items
        self.assertGreater(len(os.listdir(model.hmm_folder)), 1)

    def test_fit(self):
        """ Testing if fit tool is able to run and fit data """

        # Data matrix
        outcomes = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
        data = []
        for question_id, outcome in enumerate(outcomes):
            data.append([outcome, 0, question_id])

        # Q matrix
        q_matrix = np.array([[1, 0]]*len(outcomes))
        q_matrix[[0, 1], 1] = 1

        # Instantiate model
        model = bkt.BKT()
        model.fit(data, q_matrix)

        # Make sure self.params variable is complete after this test
        self.assertIsNotNone(model.params)

    def test_params_format(self):
        """ Testing if fitted params are on the expected format """
        # Data matrix
        outcomes = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
        data = []
        for question_id, outcome in enumerate(outcomes):
            data.append([outcome, 0, question_id])

        # Q matrix
        q_matrix = np.array([[1, 0]]*len(outcomes))
        q_matrix[[0, 1], 1] = 1

        # Instantiate model
        model = bkt.BKT()
        model.fit(data, q_matrix)

        params = model.get_params()

        for key, value in self.PARAMS_KEYS.items():
            # Assert that all keys are there
            self.assertIn(key, params)

            # Assert key type is correct
            self.assertTrue(isinstance(params[key], value))

        for skill_param in params["skills"]:
            for key, value in self.SKILLS_KEYS.items():
                # Assert that all keys are there
                self.assertIn(key, skill_param)

                # Assert key type is correct
                self.assertTrue(isinstance(skill_param[key], value))

    # def test_params_values(self):
    #     """ Testing if fitted params returned the expected values """
    #     # Data matrix
    #     outcomes = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
    #     data = []
    #     for question_id, outcome in enumerate(outcomes):
    #         data.append([outcome, 0, question_id])

    #     # Q matrix
    #     q_matrix = np.array([[1, 0]]*len(outcomes))
    #     q_matrix[[0, 1], 1] = 1

    #     # Instantiate model
    #     model = BKT()
    #     model.fit(data, q_matrix)

    #     params = model.get_params()
    #     expected_params = {
    #         'skills': [
    #             {'skill': '0',
    #              'priors': np.array([0.0000000000, 1.0000000000]),
    #              'transitions': np.array([1.0000000000, 0.0000000000,
    #                                       0.1667616137, 0.8332383863]),
    #              'emissions': np.array([0.9995594072, 0.0004405928,
    #                                     0.0003857303, 0.9996142697])},
    #             {'skill': '1',
    #              'priors': np.array([0.0000068251, 0.9999931749]),
    #              'transitions': np.array([1.0000000000, 0.0000000000,
    #                                       0.0013481800, 0.9986518200]),
    #              'emissions': np.array([0.7000000000, 0.3000000000,
    #                                     0.0000000000, 1.0000000000])}],
    #         'SolverId': '1.1', 'nK': '2', 'nG': '1', 'nS': '2', 'nO': '2',
    #         'nZ': '1', 'Null skill ratios': '  1.0000000\t  0.0000000'}

    #     self.assertEqual(params, expected_params)

    def test_predict_proba(self):
        """ Testing probability predict function """
        # Data matrix
        outcomes = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
        data = []
        for question_id, outcome in enumerate(outcomes):
            data.append([outcome, 0, question_id])

        # Q matrix
        q_matrix = np.array([[1, 0]]*len(outcomes))
        q_matrix[[0, 1], 1] = 1

        # Instantiate model
        model = bkt.BKT()
        model.fit(data, q_matrix)

        # Data for predicting should contain only one student
        data = []
        for question_id, outcome in enumerate(outcomes):
            data.append([outcome, question_id])
        y_pred_proba = model.predict_proba(data, q_matrix)

        # Assert y_pred_proba has correct data type
        self.assertTrue(isinstance(y_pred_proba, np.ndarray))
        for row in y_pred_proba:
            self.assertTrue(isinstance(row, np.ndarray))
            for element in row:
                self.assertTrue(isinstance(element, float))

        # Assert y_pred_proba is in correct shape
        self.assertTrue(y_pred_proba.shape, (len(data), 2))

    def test_predict(self):
        """ Testing predict function """
        # Data matrix
        outcomes = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
        data = []
        for question_id, outcome in enumerate(outcomes):
            data.append([outcome, 0, question_id])

        # Q matrix
        q_matrix = np.array([[1, 0]]*len(outcomes))
        q_matrix[[0, 1], 1] = 1

        # Instantiate model
        model = bkt.BKT()
        model.fit(data, q_matrix)

        # Data for predicting should contain only one student
        data = []
        for question_id, outcome in enumerate(outcomes):
            data.append([outcome, question_id])

        y_pred = model.predict(data, q_matrix)

        # Assert y_pred has correct data type
        self.assertTrue(isinstance(y_pred, np.ndarray))
        for outcome in y_pred:
            self.assertTrue(isinstance(outcome, np.int64))

        # Assert y_pred is in correct shape
        self.assertTrue(y_pred.shape, (len(data),))

    def test_score(self):
        # Data matrix
        outcomes = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]
        data = []
        for question_id, outcome in enumerate(outcomes):
            data.append([outcome, 0, question_id])

        # Q matrix
        q_matrix = np.array([[1, 0]]*len(outcomes))
        q_matrix[[0, 1], 1] = 1

        # Instantiate model
        model = bkt.BKT()
        model.fit(data, q_matrix)

        # Data for predicting should contain only one student
        data = []
        for question_id, outcome in enumerate(outcomes):
            data.append([outcome, question_id])

        model.predict(data, q_matrix)
        scores = model.score()

        # Assert that all 4 scores are coming
        self.assertTrue(len(scores), 4)

        # Assert scores types
        for value in scores:
            self.assertTrue(isinstance(value, float))

        # Assert scores values for current example
        expected_scores = (36.685475118198525, 44.651333306630455,
                           0.3009647032927156, 0.9)
        for idx in range(len(scores)):
            self.assertAlmostEqual(scores[idx], expected_scores[idx])


if __name__ == '__main__':
    unittest.main()
