#   Copyright 2018 SciNet (https://github.com/eth-nn-physics/nn_physical_concepts)
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
#   Modifications copyright (C) 2020 (Samuel Smith [samuel.smith@nottingham.ac.uk] at the University of Nottingham, UK).

import os

file_path = os.path.dirname(__file__)
data_path = os.path.realpath(os.path.join(file_path, '../data/')) + '/'
tf_save_path = os.path.realpath(os.path.join(file_path, '../tf_save/')) + '/'
tf_log_path = os.path.realpath(os.path.join(file_path, '../tf_log/')) + '/'
