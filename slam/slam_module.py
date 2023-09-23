#!/usr/bin/env python3

from pipeline.pipeline_module import MIMOPipelineModule
import numpy as np
import copy

class SlamModule(MIMOPipelineModule):
    def __init__(self, name, args, device="cpu"):
        super().__init__(name, args.parallel_run, args)
        self.device = device
        self.outpath = args.outpath

    def spin_once(self, input):
        output = self.slam(input)
        if output[1] is not None:
            if self.device != "cpu":
                self.cam0_poses = copy.copy(output[1]['cam0_poses']).detach().cpu().numpy()
            else:
                self.cam0_poses = copy.copy(output[1]['cam0_poses']).detach().numpy()
        if not output or self.slam.stop_condition():
            super().shutdown_module()
        return output
 
    def initialize_module(self):
        if self.name == "VioSLAM":
            from slam.vio_slam import VioSLAM
            self.slam = VioSLAM(self.name, self.args, self.device)
        else:
            raise NotImplementedError
        return super().initialize_module()

    def save(self):
        if self.outpath is not None:
            np.save(self.outpath+'/final_cam_poses.npy', self.cam0_poses)
        else:
            np.save('final_cam_poses.npy')
