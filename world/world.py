from world.components.gelsight.gelsight2014 import GelSight2014
from yarok.comm.worlds.empty_world import EmptyWorld
from yarok import Platform, PlatformMJC, PlatformHW, Injector, component, ConfigBlock

from .components.geltip.geltip import GelTip
from .components.tumble_tower.tumble_tower import TumbleTower
from .components.ur5e.ur5e import UR5e
from .components.robotiq_2f85.robotiq_2f85 import Robotiq2f85
from .components.cam.cam import Cam
# from world.shared.colors import colors_names, color_map

import os
__location__ = os.path.dirname(os.path.abspath(__file__))


@component(
    extends=EmptyWorld,
    components=[
        GelTip,
        GelSight2014,
        TumbleTower,
        UR5e,
        Robotiq2f85,
        Cam
    ],
    defaults={
        'dataset_name': 'wear'
    },
    template="""
        <mujoco>
            <option impratio="50" noslip_iterations="15"/>
            <asset>
                <texture type="skybox" 
                    file="assets/robot_lab.png"
                    rgb1="0.6 0.6 0.6" 
                    rgb2="0 0 0"/>
                <texture 
                    name="wood_texture"
                    type="cube" 
                    file="assets/white_wood.png"
                    width="400" 
                    height="400"/>

                <mesh name="mount" 
                      file="assets/{dataset_name[0:4]}_mount.stl" 
                      scale="0.001 0.001 0.001"/>
                <mesh name="knife" 
                      file="assets/knife.stl" 
                      scale="0.001 0.001 0.001"/>
                <mesh name="screw" 
                      file="assets/screw.stl" 
                      scale="0.001 0.001 0.001"/>
                      
                <hfield name="sp_hfield" file="assets/heightmap.png" size="0.005 0.01 0.001 0.001"/>


                <material name="wood" texture="wood_texture" specular="0.1"/>
                <material name="gray_wood" texture="wood_texture" rgba="0.6 0.4 0.2 1" specular="0.1"/>
                <material name="white_wood" texture="wood_texture" rgba="0.6 0.6 0.6 1" specular="0.1"/>
                <material name="brown_wood"  rgba="0.5 0.4 0.3 1" specular="0.1"/>
                <material name="black_metal" rgba="0.05 0.05 0.05 1" specular="1.0"/>
                <material name="gray_metal" rgba="0.85 0.85 0.85 1" specular="1.0"/>
                <material name="black_plastic" rgba="0.15 0.15 0.15 1" specular="0.1"/>
            </asset>
            <default>
                <!-- <default class='pp-block'>
                     <geom type="box" 
                           size="{bs} {bs} {bs}"
                           mass="0.0001"
                           material="wood"
                           zaxis="0 1 0"/>
                </default> -->
            </default>
            <worldbody>
                <!-- 
                <light directional="true" 
                    diffuse="{color_map(light, 0.1)}" 
                    specular="{color_map(light, 0.1)}" 
                    pos="1.0 1.0 5.0" 
                    dir="0 -1 -1"/>
                -->
                <body pos="{0.3 + p_cam[0]*0.1} -1.3 {0.6 + p_cam[0]*0.1}" euler="1.57 -3.14 0">
                    <cam name="cam" />
                </body> 
                

               <body>
                    <!-- wall -->
                    <geom type="box" pos="0 0.45 0" size="0.8 0.05 1.0" material="white_wood"/>
                    
                    <!-- table base -->
                    <geom type="box" pos="0 0 0.18" size="1.0 0.5 0.1" material="white_wood"/>
                    
                    <!-- black plate -->
                    <geom type="box" pos="0.5 -0.3 0.285" size="0.2 0.2 0.005" material="black_metal"/> 
                    
                    <!-- wood board -->
                    <geom type="box" pos="-0.1 -0.1 0.29" size="0.2 0.2 0.01" material="brown_wood"/> 
                    
                    <!-- wear and tear tool mount -->
                    <geom type="mesh" mesh="mount" material="black_plastic" pos="0.5195 -0.42 0.288" zaxis="0 -1 0"/>
                    
                    <!-- wear sand paper -->
                    <!-- #1 is used to patch yarok's missing renaming hfields -->
                    <geom if="dataset_name == 'wear'" 
                        name="sand_paper" 
                        type="hfield" 
                        hfield="#1:sp_hfield" 
                        pos="0.53 -0.46 0.336" material="black_metal"/>
                    
                    <!-- tear tools -->
                    <geom if="dataset_name == 'tear_knife'" type="mesh" mesh="knife" 
                        material="gray_metal" pos="0.5195 -0.455 0.288" zaxis="0 -1 0" /> 
                    <geom if="dataset_name == 'tear_screw'" type="mesh" mesh="screw" 
                        material="gray_metal" pos="0.5195 -0.455 0.288" zaxis="0 -1 0" /> 
               </body> 

                <body euler="0 0 3.14" pos="0 -0.2 0.3">
                    <ur5e name='arm'>
                       <robotiq-2f85 name="gripper" left_tip="{False}" right_tip="{False}" parent="ee_link"> 
                          <body euler="1.57079 0 -1.57079" parent="left_tip">
                            <body pos="-0.07 0 0">
                                <gelsight2014 name="left_geltip" label_color="255 0 0"/>
                            </body>
                           </body>
                        </robotiq-2f85> 
                    </ur5e> 
                </body>
            </worldbody>
        </mujoco>
    """
)
class WearNTearWorld:
    pass


def shared_config(behaviour, env, **kwargs):
    return {
        'world': WearNTearWorld,
        'behaviour': behaviour,
        'defaults': {
            'environment': env,  # update this to sim for running in sim environment.
            'behaviour': kwargs,
            'components': {
                '/': kwargs
            }
        },
        'environments': {
            'sim': {
                'platform': {
                    'class': PlatformMJC,
                    'interfaces': {
                        '/arm': {
                            # it seems that there's a bug in yarok,
                            # so I've repeated in the interface
                            'initial_position': ([-0.44, -0.44, 0.3], [0.0, 1.6, 0.0])
                        }
                    }
                },
            },
            'real': {
                'platform': {
                    'class': PlatformHW,
                    'interfaces': {
                        '/left_geltip': {
                            'cam_id': 2
                        },
                        '/cam': {
                            'cam_id': 4
                        }
                    }
                },

            }
        }
    }
