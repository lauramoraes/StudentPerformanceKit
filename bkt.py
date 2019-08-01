import requests
import zipfile
import subprocess
import os
import uuid
import shlex

algorithm_lookup = {
    "bw": "1.1",
    "gd": "1.2",
    "cgd_pr": "1.3.1",
    "cgd_fr": "1.3.2",
    "cgd_hs": "1.3.3"
}


class BKT(object):
    def __init__(self,
                 hmm_folder='hmm-scalable-818d905234a8600a8e3a65bb0f7aa4cf06423f1a',
                 git_commit='818d905234a8600a8e3a65bb0f7aa4cf06423f1a'):

        # Git commit to download hmm-scalable
        self.git_commit = git_commit
        # Set HMM-scalable folder.
        self.hmm_folder = hmm_folder
        # Params
        self.params = None

    def _create_data_file(self, data, q_matrix):
        if not os.path.exists("hmm_files"):
            os.makedirs("hmm_files")
        filename = "hmm_files/%s" % uuid.uuid4().hex

        # Create data file in the format expected by the tool
        with open("%s.txt" % filename, "w") as step_file:
            for row in data:
                outcome, student_id, question_id = row
                # Transform correct to 1 and incorrect to 2 (requred by the tool)
                outcome = -outcome+2
                skills = np.where(q_matrix[question_id] == 1)
                skills = "~".join(str(skill) for skill in skills[0])
                step_file.write("%s\t%s\t%s\t%s\n" % (outcome, student_id, question_id, skills))
        return filename

    def download(self):
        """  This implementation is a wrapper around the
        HMM-scalable tool ( http://yudelson.info/hmm-scalable).
        This function will download and install the original implementation.

        Returns
        -------
        self: object

        Notes
        -----
        This is a wrapper around the HMM-scalable tool (http://yudelson.info/hmm-scalable).
        """

        # Download zipfile from GitHub
#         results = requests.get('https://github.com/myudelson/hmm-scalable/archive/master.zip')
        results = requests.get('https://github.com/myudelson/hmm-scalable/archive/%s.zip' % self.git_commit)
        with open('/tmp/hmm-scalable.zip', 'wb') as f:
            f.write(results.content)

        # Extract zipfile
        file = zipfile.ZipFile('/tmp/hmm-scalable.zip')
        file.extractall(path='.')

        # Install
        process = subprocess.Popen("make all", stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=self.hmm_folder, shell=True)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise RuntimeError("Could not build HMM tool. Check if the make utility is installed "
                               "and if the folder has appropriate permissions.\n "
                               "Code: %d\n"
                               "Error: %s" % (process.returncode, stderr))

    def fit(self, data, q_matrix, solver='bw', iterations=200):
        """ Fit BKT model to data.
        As of July 2019, just default parameters are allowed.

        Parameters
        ----------
        data : {array-like}, shape (n_steps, 3)
            Sequence of students steps. Each of the three dimensions are:
            Observed outcome: 0 for fail and 1 for success
            Student id: student unique identifier
            Question id: question id in q_matrix

        q_matrix: matrix, shape (n_questions, n_concepts)
            Each row is a question and each column a concept.
            If the concept is present in the question, the
            correspondent cell should contain 1, otherwise, 0.

        solver: string, optional
            Algorithm used to fit the BKT model. Available solvers are:
            'bw': Baum-Welch (default)
            'gd': Gradient Descent
            'cgd_pr': Conjugate Gradient Descent (Polak-Ribiere)
            'cgd_fr': Conjugate Gradient Descent (Fletcher–Reeves)
            'cgd_hs': Conjugate Gradient Descent (Hestenes-Stiefel)

        iterations: integer, optional
            Maximum number of iterations

        Returns
        -------
        self: object

        Notes
        -----
        This is a wrapper around the HMM-scalable tool (http://yudelson.info/hmm-scalable)
        """
        # Create data file
        filename = self._create_data_file(data, q_matrix)

        # Run train program
        command = "./trainhmm -s %s -i %d -d ~ ../%s.txt ../%s_model.txt" % (
            algorithm_lookup[solver], iterations, filename, filename)
        args = shlex.split(command)
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=self.hmm_folder)
        process.wait()
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise RuntimeError("Could not train HMM model. Check if the HMM files are properly created and "
                               "accessible.\n"
                               "Code: %d\n"
                               "Error: %s" % (process.returncode, stderr))

        # Extract fitted params
        with open("%s_model.txt" % filename, "r") as model_file:
            content = model_file.read()
        params = {"skills": []}
        params_regex = re.findall(r'^([\w ]+)\t(.*)$', content, flags=re.M)
        idx = 0
        while idx < len(params_regex):
            param, value = params_regex[idx]
            # Get skills. They start at zero.
            try:
                skill = int(param)
                params["skills"].append({
                    "skill": value,
                    # Get PI matrix
                    "priors": np.asarray([float(i) for i in params_regex[idx+1][1].split("\t")]),
                    # Get A matrix
                    "transitions": np.asarray([float(i) for i in params_regex[idx+2][1].split("\t")]),
                    # Get B matrix
                    "emissions": np.asarray([float(i) for i in params_regex[idx+3][1].split("\t")])
                })
                idx += 4
            except ValueError:
                params[param] = value
                idx += 1

        self.params = params
        return self

    def _predict(self, data, q_matrix, model_file=None):
        """ Predict student outcomes based on trained model.

        Parameters
        ----------
        data : {array-like}, shape (n_steps, 3)
            Sequence of students steps. Each of the three dimensions are:
            Observed outcome: 0 for fail and 1 for success
            Student id: student unique identifier
            Question id: question id in q_matrix

        Returns
        -------
        outcome : array, shape (n_steps,)
            Predicted outcomes for steps in data

        Notes
        -----
        This is a wrapper around the HMM-scalable tool (http://yudelson.info/hmm-scalable)
        """
        PARAMS_ORDER = ["SolverId", "nK", "nG", "nS", "nO", "nZ", "Null skill ratios"]

        # Create data file
        filename = self._create_data_file(data, q_matrix)

        # If model file does not exists, create one.
        if not model_file:
            model_file = "%s_model.txt" % filename

            # Create model file from set params
            with open(model_file, "w") as param_file:
                model_params = self.get_params()

                # Write general parameters to file
                for param in PARAMS_ORDER:
                    param_file.write("%s\t%s\n" % (param, model_params[param]))

                # Write skill parameters to file
                for idx, skill in enumerate(model_params["skills"]):
                    priors = "\t".join(["%.10f" % element for element in skill["priors"]])
                    transitions = "\t".join(["%.10f" % element for element in skill["transitions"]])
                    emissions = "\t".join(["%.10f" % element for element in skill["emissions"]])
                    skill_text = "%d\t%s\nPI\t%s\nA\t%s\nB\t%s\n" % (
                        idx, skill["skill"], priors, transitions, emissions)
                    param_file.write(skill_text)

        # Run predict program
        command = "./predicthmm -p 1 -d ~ ../%s.txt ../%s_model.txt ../%s_predict.txt" % (
            filename, filename, filename)
        args = shlex.split(command)
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=self.hmm_folder)
        process.wait()
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            raise RuntimeError("Could not predict HMM model. Check if the HMM files are properly created and "
                               "accessible.\n"
                               "Code: %d\n"
                               "Error: %s" % (process.returncode, stderr))

        # Read from predict file
        with open("%s_predict.txt" % filename, "r") as predict_file:
            y_pred_txt = predict_file.read().strip().split("\n")
            y_pred_proba = []
            for row in y_pred_txt:
                y_pred_proba.append([float(i) for i in row.split("\t")])

        # Swap columns so column 0 is outcome 0 (incorrect) and column 1 is outcome 1 (correct)
        y_pred_proba = np.asarray(y_pred_proba)
        y_pred_proba_tmp = y_pred_proba.copy()
        y_pred_proba[:,0] = y_pred_proba_tmp[:,1]
        y_pred_proba[:,1] = y_pred_proba_tmp[:,0]

        return y_pred_proba

    def predict_proba(self, data, q_matrix, model_file=None):
        """ Predict student outcome probabilities based on trained model.

        Parameters
        ----------
        data : {array-like}, shape (n_steps, 3)
            Sequence of students steps. Each of the three dimensions are:
            Observed outcome: 0 for fail and 1 for success
            Student id: student unique identifier
            Question id: question id in q_matrix

        Returns
        -------
        outcome : {array-like}, shape (n_steps, 2)
            Outcome probabilites for steps in data. Column 0 corresponds to outcome 0 (incorrect)
            and column 1 to outcome 1 (correct)

        Notes
        -----
        This is a wrapper around the HMM-scalable tool (http://yudelson.info/hmm-scalable)
        """
        y_pred_proba = self._predict(data, q_matrix, model_file)
        return y_pred_proba

    def predict(self, data, q_matrix, model_file=None):
        """ Predict student outcomes based on trained model. This is just the hard-assigment
        (highest probability) of the outcome probabilities.

        Parameters
        ----------
        data : {array-like}, shape (n_steps, 3)
            Sequence of students steps. Each of the three dimensions are:
            Observed outcome: 0 for fail and 1 for success
            Student id: student unique identifier
            Question id: question id in q_matrix

        Returns
        -------
        outcome : {array-like}, shape (n_steps,)
            Outcomes for steps in data. Outcome 0 is incorrect and outcome 1 is correct.

        Notes
        -----
        This is a wrapper around the HMM-scalable tool (http://yudelson.info/hmm-scalable)
        """
        y_pred_proba = self._predict(data, q_matrix, model_file)
        y_pred = np.argmax(y_pred_proba, axis=1)
        return y_pred

    def get_params(self):
        """ Get fitted params.

        Returns
        -------
        params : list. List containing the prior, transition and emission values for each skill.
        """
        if self.params is None:
            raise RuntimeError("You should run fit before getting params")
        return self.params

    def set_params(self, params):
        """ Set model params. No validation is done for this function.
        Make sure the params variable is in the expected format.

        Returns
        -------
        self: object
        """
        self.params = params
        return self
