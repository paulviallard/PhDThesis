import os
import re
import time
import itertools
import argparse
import configparser
import subprocess
import logging

import numpy as np
import math
###############################################################################

class LaunchConfig():

    def __init__(self, f):
        # We set the prefix of the job names (useful for slurm ...)
        self.prefix_run_name = "job_{}_".format(int(time.time()))

        # We read the configuration file (.ini)
        config = configparser.ConfigParser(empty_lines_in_values=False)
        config.read(f)

        # We initialize the root of a tree. A node is organized as follows:
        # 1- Name of the jobs (name of the node)
        # 2- Name of the parent jobs (name of the parent node)
        # 3- "Data" of the job (i.e., hyperparameters, variables, command)
        # 4- Sub-tree containing the jobs that need to be run for every set of
        # hyperparameters
        # 5- Sub-tree containing the jobs that need to be run after the the
        # previous sub-tree
        job_tuple = (None, None, None, [], [])
        self.job_tree = job_tuple

        # We initialize a dict to insert easily in the tree
        job_dict = {None: job_tuple}

        # We initialize the previous node that we seen
        prog_previous = None

        # For every node of the tree (the sections in the ini file)
        for job_name in config.sections():

            # We remove the space in the sections of the ini file
            job_name_list = job_name.replace(" ", "")
            job_name_list = re.split("(-(?:-|>))", job_name_list)

            # If the section can be rewriten as "[someth]"
            if(len(job_name_list) == 1):

                # We initialize the node of the tree with its "data"
                job_tuple = (job_name_list[0], prog_previous,
                             config[job_name], [], [])

                # We insert the new node in the dict
                job_dict[job_name_list[0]] = job_tuple
                # We insert the new node in the parent node
                job_dict[prog_previous][3].append(job_tuple)
                # We keep this node
                prog_previous = job_name_list[0]

            elif(len(job_name_list) == 3):

                # If the section can be rewriten as "[-> someth]"
                if(job_name_list[0] == ""):
                    # We initialize the node of the tree with its "data"
                    job_tuple = (job_name_list[2], prog_previous,
                                 config[job_name], [], [])
                    # We insert the new node in the dict
                    job_dict[job_name_list[2]] = job_tuple
                    # We insert the new node in the parent node
                    job_dict[prog_previous][3].append(job_tuple)

                # If the section can be rewriten as "[someth1 -> someth2]"
                elif(job_name_list[1] == "->"):
                    # We initialize the node of the tree with its "data"
                    job_tuple = (job_name_list[2], job_name_list[0],
                                 config[job_name], [], [])
                    # We insert the new node in the dict
                    job_dict[job_name_list[2]] = job_tuple
                    # We insert the new node in the node indicated by "someth1"
                    job_dict[job_name_list[0]][4].append(job_tuple)

                # If the section can be rewriten as "[someth1 -- someth2]"
                elif(job_name_list[1] == "--"):
                    # We get the parent of the node "someth1"
                    job_parent = job_dict[job_name_list[0]][1]
                    # We initialize the node of the tree with its "data"
                    job_tuple = (job_name_list[2], job_parent,
                                 config[job_name], [], [])
                    # We insert the new node in the dict
                    job_dict[job_name_list[2]] = job_tuple
                    # We insert the new node in the parent node that is
                    # associated to "someth1"
                    job_dict[job_parent][3].append(job_tuple)

                # We keep this node
                prog_previous = job_name_list[2]

    def _get_param_list(self, param_list, known_param):

        # We get the params (, the variables and the command...)
        param_list = dict(param_list)

        # We remove the command
        if("command" in param_list):
            del param_list["command"]
        # We remove the variables
        for key in list(param_list.keys()):
            if(re.match("[$]{[^}]+}", key)):
                del param_list[key]

        # We create the python variable to use parameters the ini file
        for param_name in known_param:
            exec(param_name+" = known_param['"+param_name+"']")

        # We create a todo list (see further)
        todo_param_list = {}

        # For each params
        for param_name in list(param_list.keys()):

            try:
                # We interpret the param
                exec("param_list['" + param_name + "'] = "
                     + param_list[param_name])
            except NameError:
                # If we cannot interpret the param, we add the param in the
                # todo list
                todo_param_list[param_name] = param_list[param_name]
                del param_list[param_name]
                continue

            # If the param is a dict, we need to transform it in a list of
            # params
            if(isinstance(param_list[param_name], dict)):

                # We get the dict
                param_dict = param_list[param_name]

                # If the elements of the dict are not a list, we create a list
                # with the element
                for key in param_dict:
                    if(not(isinstance(param_dict[key], list))):
                        param_dict[key] = [param_dict[key]]

                # We inialize a list
                param_str_list = []
                # For each combination of params in the dict
                for param in itertools.product(*param_dict.values()):
                    param = dict(zip(param_dict.keys(), param))

                    # We create a string with the params
                    param_str = ""
                    for key in sorted(param.keys()):
                        if(not(isinstance(param[key], str))):
                            param_str += key+"="+str(param[key])+","
                        else:
                            param_str += key+"=\""+str(param[key])+"\","
                    param_str = param_str[:-1]

                    # and we add it to the list
                    param_str_list.append(param_str)

                # We replace the dict by the list
                param_list[param_name] = param_str_list

            # If the parameter type is not a list
            if(not(isinstance(param_list[param_name], list))):
                # We create a list with the element
                param_list[param_name] = [param_list[param_name]]

        return param_list, todo_param_list

    def _get_var_list(self, param_list, known_param, known_var):

        # We get the variables (i.e., the params and the command...)
        var_list = dict(param_list)
        # We remove the params and the command
        for key in list(var_list.keys()):
            if(not(re.match("[$]{[^}]+}", key))):
                del var_list[key]

        # We update the list of variables with the previous ones
        known_var = dict(known_var)
        known_var.update(var_list)
        var_list = known_var

        # We add the params in the variables
        new_known_param = {}
        for param in known_param:
            new_known_param["${"+param+"}"] = str(known_param[param])
        known_var.update(new_known_param)

        # We create two special variables: ${path} and ${params}
        # ${path} contains a path containing the current hyperparams
        # ${params} contains a list of params for the run
        var_list["${path}"] = ""
        var_list["${params}"] = ""
        for arg_name in sorted(list(known_param)):
            var_list["${path}"] += "{}={}/".format(
                arg_name, known_param[arg_name])
            var_list["${params}"] += " --{}={}".format(
                arg_name, known_param[arg_name])
        var_list["${path}"] = var_list["${path}"][:-1]

        # While the replacements are not stabilized
        terminated = False
        while not(terminated):
            terminated = True
            for var_name in var_list:
                for var_name_replace in var_list:
                    # We replace the name of the variables by their values
                    old_var_list = str(var_list[var_name])
                    var_list[var_name] = var_list[var_name].replace(
                        var_name_replace, var_list[var_name_replace])
                    if(old_var_list != var_list[var_name]):
                        terminated = False

        return var_list

    def _get_command(self, job_tree):
        # We get the command of the current job
        if("command" in job_tree[2]):
            return job_tree[2]["command"]

    def _construct_command(self, command, var_list):
        # For each variable
        for var_name in var_list:
            # We replace the variable by its content in the command
            command = command.replace(var_name, var_list[var_name])
        return command

    def run(self):
        # We initialize a run index (useful for slurm ...)
        self.__run_index = 0
        # We run the jobs of the subtrees
        for prog_subtree in self.job_tree[3]:
            self._run(prog_subtree, prog_subtree[2], {}, {}, [])

    def _run(self, job_tree, param_list,
             known_param, known_var, known_dependency
             ):

        # We keep the old parameters to check if there is a problem
        old_param_list = param_list

        # We get the list of the hyperparameters (of a given job)
        param_list, todo_param_list = self._get_param_list(
            param_list, known_param)

        # We initalize a list of run dependencies
        known_dependency = sorted(list(set(known_dependency)))
        dependency_1 = list(known_dependency)

        # For each combination of params
        for param in itertools.product(*param_list.values()):

            # We check if there is still some parameters to interpret
            if(len(todo_param_list) > 0):
                known_param_ = {
                    key: value for key, value in zip(param_list.keys(), param)}

                # If we cannot interpret some parameters, there is a problem
                if(len(old_param_list) == len(todo_param_list)):
                    s_error = ', '.join(todo_param_list.keys())
                    if(len(todo_param_list) == 1):
                        s_error = "The parameter " + s_error + " is"
                    else:
                        s_error = "The parameters " + s_error + " are"
                    s_error += " not correct"
                    raise RuntimeError(s_error)

                # We update the known parameters before interpreting
                # recursively
                known_param_.update(known_param)

                # We run recursively the algorithm with the interpreted
                # parameters (to run the new loops ...)
                self._run(
                    job_tree, todo_param_list, known_param_,
                    known_var, known_dependency)
                continue

            # We increase the run index
            self.__run_index += 1
            # We create a name of this run
            run_name = self.prefix_run_name+str(self.__run_index)

            # We update our params for a given run
            param_ = dict(known_param)
            param_.update(dict(zip(param_list.keys(), param)))

            # We update the variables for a given run
            var_ = dict(known_var)
            var_.update(self._get_var_list(job_tree[2], param_, var_))

            # We get the command
            command = self._get_command(job_tree)

            # If the command exists
            if(command is not None):
                # We interpret the command (with the variables)
                command = self._construct_command(command, var_)
                # We run the command
                self._run_command(command, run_name, known_dependency)

            # For each subtree (of type "[-> someth]")
            for job_subtree in job_tree[3]:
                # We run recursively the subtree (and we get the dependency)
                dependency_ = self._run(
                    job_subtree, job_subtree[2], param_,
                    var_, known_dependency+[run_name])
                # We add the new dependences in the list
                dependency_1 = sorted(list(set(dependency_1 + dependency_)))

            # We add our current run in the (first) dependency list
            dependency_1.append(run_name)

        # We create a second (and final) dependency list
        dependency_2 = list(dependency_1)

        # For each subtree (of type "[someth1 -> someth2]")
        for job_subtree in job_tree[4]:
            # We run recursively the subtree (and we get the dependency)
            dependency_ = self._run(
                job_subtree, job_subtree[2], {}, {}, dependency_1)
            # We add the new dependencies in the list
            dependency_2 = sorted(list(set(dependency_2 + dependency_)))

        # We return the dependencies of the node
        return dependency_2

    def _run_command(self, command, run_name=None, job_list=None):
        # To be implemented !
        raise NotImplementedError


###############################################################################

class PrintLaunchConfig(LaunchConfig):

    def _run_command(self, command, run_name=None, dependency=None):
        # We create the message
        msg = "Job: {}".format(run_name)
        msg += " - Dependency: {}".format(", ".join(dependency))

        # We print the message
        logging.info(msg+"\n")
        logging.info("-"*len(msg)+"\n")

        # We print the command
        logging.info(command+"\n")


###############################################################################

class SequentialLaunchConfig(LaunchConfig):

    def _run_command(self, command, run_name=None, job_list=None):
        # We execute the command in the shell
        subprocess.call(command, shell=True)


###############################################################################

class SlurmLaunchConfig(LaunchConfig):

    def get_job_id(self, run_name):
        # We execute in the shell a command to get the id of a job given the
        # job name
        job_id = subprocess.run(
            "squeue -n \"{}\" -o \"%A\" 2> /dev/null | tr \"\n\" \" \"".format(
                run_name), stdout=subprocess.PIPE, universal_newlines=True,
            shell=True).stdout
        # We get the job id by stdout
        job_id = job_id.replace("JOBID ", "")
        if(job_id != ""):
            job_id = int(job_id)
        else:
            job_id = None
        return job_id

    def _run_command(self, command, run_name=None, job_list=None):

        new_job_list = []
        if(job_list is not None):
            for run_name_ in job_list:
                job_id = self.get_job_id(run_name_)
                if(job_id is not None):
                    new_job_list.append(job_id)
        job_list = new_job_list

        # We get all the job id on which our job depends
        if(len(job_list) > 0):
            dependency = "--dependency=afterok:"
        else:
            dependency = ""
        for job_id in job_list:
            if(job_id is not None):
                dependency += str(job_id)+":"
        dependency = dependency[:-1]

        # We run the command to run slurm
        subprocess.call("sbatch {} -J {} ./run_slurm '{}'".format(
            dependency, run_name, command
        ), shell=True)


###############################################################################

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    #  logging.getLogger().disabled = True
    logging.StreamHandler.terminator = ""

    # ----------------------------------------------------------------------- #

    arg_parser = argparse.ArgumentParser(
        description='Execute the job described in the ini file')

    arg_parser.add_argument(
        "mode", metavar="mode", type=str,
        help="The mode of execution (either print, sequential or slurm)")
    arg_parser.add_argument(
        "path", metavar="path", type=str,
        help="The path of the ini file")

    arg_list = arg_parser.parse_args()
    mode = arg_list.mode
    path = arg_list.path

    if(mode != "print" and mode != "sequential" and mode != "slurm"):
        arg_parser.error("The mode of execution must be"
                         + " either print, sequential or slurm")
    if(not(os.path.exists(path))):
        arg_parser.error("The path" + path + " does not exist")

    # ----------------------------------------------------------------------- #

    if(mode == "print"):
        PrintLaunchConfig(path).run()
    elif(mode == "sequential"):
        SequentialLaunchConfig(path).run()
    elif(mode == "slurm"):
        SlurmLaunchConfig(path).run()
