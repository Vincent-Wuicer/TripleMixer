# Copyright 2024 - xiongwei zhao @ grandzhaoxw@gmail.com
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from .pc_dataset import Collate
from .snow_wads import SnowWads
from .snow_kitti import SnowKITTI
from .fog_kitti import FogKITTI
from .rain_kitti import RainKITTI
from .snow_nus import SnowNus
from .fog_nus import FogNus


__all__ = [SnowWads, SnowKITTI, FogKITTI, RainKITTI, SnowNus, FogNus, Collate]
LIST_DATASETS = {"snow_wads": SnowWads, "snow_kitti": SnowKITTI, "fog_kitti": FogKITTI, 
                 "rain_kitti": RainKITTI, "snow_nus": SnowNus, "fog_nus": FogNus}
