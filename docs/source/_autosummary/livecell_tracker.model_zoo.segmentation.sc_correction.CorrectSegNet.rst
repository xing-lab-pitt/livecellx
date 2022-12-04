livecell\_tracker.model\_zoo.segmentation.sc\_correction.CorrectSegNet
======================================================================

.. currentmodule:: livecell_tracker.model_zoo.segmentation.sc_correction

.. autoclass:: CorrectSegNet

   
   .. automethod:: __init__

   
   .. rubric:: Methods

   .. autosummary::
   
      ~CorrectSegNet.__init__
      ~CorrectSegNet.add_module
      ~CorrectSegNet.add_to_queue
      ~CorrectSegNet.all_gather
      ~CorrectSegNet.apply
      ~CorrectSegNet.backward
      ~CorrectSegNet.bfloat16
      ~CorrectSegNet.buffers
      ~CorrectSegNet.children
      ~CorrectSegNet.clip_gradients
      ~CorrectSegNet.configure_callbacks
      ~CorrectSegNet.configure_gradient_clipping
      ~CorrectSegNet.configure_optimizers
      ~CorrectSegNet.configure_sharded_model
      ~CorrectSegNet.cpu
      ~CorrectSegNet.cuda
      ~CorrectSegNet.double
      ~CorrectSegNet.eval
      ~CorrectSegNet.extra_repr
      ~CorrectSegNet.float
      ~CorrectSegNet.forward
      ~CorrectSegNet.freeze
      ~CorrectSegNet.get_buffer
      ~CorrectSegNet.get_extra_state
      ~CorrectSegNet.get_from_queue
      ~CorrectSegNet.get_parameter
      ~CorrectSegNet.get_progress_bar_dict
      ~CorrectSegNet.get_submodule
      ~CorrectSegNet.half
      ~CorrectSegNet.ipu
      ~CorrectSegNet.load_from_checkpoint
      ~CorrectSegNet.load_state_dict
      ~CorrectSegNet.log
      ~CorrectSegNet.log_dict
      ~CorrectSegNet.log_grad_norm
      ~CorrectSegNet.lr_scheduler_step
      ~CorrectSegNet.lr_schedulers
      ~CorrectSegNet.manual_backward
      ~CorrectSegNet.modules
      ~CorrectSegNet.named_buffers
      ~CorrectSegNet.named_children
      ~CorrectSegNet.named_modules
      ~CorrectSegNet.named_parameters
      ~CorrectSegNet.on_after_backward
      ~CorrectSegNet.on_after_batch_transfer
      ~CorrectSegNet.on_before_backward
      ~CorrectSegNet.on_before_batch_transfer
      ~CorrectSegNet.on_before_optimizer_step
      ~CorrectSegNet.on_before_zero_grad
      ~CorrectSegNet.on_epoch_end
      ~CorrectSegNet.on_epoch_start
      ~CorrectSegNet.on_fit_end
      ~CorrectSegNet.on_fit_start
      ~CorrectSegNet.on_hpc_load
      ~CorrectSegNet.on_hpc_save
      ~CorrectSegNet.on_load_checkpoint
      ~CorrectSegNet.on_post_move_to_device
      ~CorrectSegNet.on_predict_batch_end
      ~CorrectSegNet.on_predict_batch_start
      ~CorrectSegNet.on_predict_dataloader
      ~CorrectSegNet.on_predict_end
      ~CorrectSegNet.on_predict_epoch_end
      ~CorrectSegNet.on_predict_epoch_start
      ~CorrectSegNet.on_predict_model_eval
      ~CorrectSegNet.on_predict_start
      ~CorrectSegNet.on_pretrain_routine_end
      ~CorrectSegNet.on_pretrain_routine_start
      ~CorrectSegNet.on_save_checkpoint
      ~CorrectSegNet.on_test_batch_end
      ~CorrectSegNet.on_test_batch_start
      ~CorrectSegNet.on_test_dataloader
      ~CorrectSegNet.on_test_end
      ~CorrectSegNet.on_test_epoch_end
      ~CorrectSegNet.on_test_epoch_start
      ~CorrectSegNet.on_test_model_eval
      ~CorrectSegNet.on_test_model_train
      ~CorrectSegNet.on_test_start
      ~CorrectSegNet.on_train_batch_end
      ~CorrectSegNet.on_train_batch_start
      ~CorrectSegNet.on_train_dataloader
      ~CorrectSegNet.on_train_end
      ~CorrectSegNet.on_train_epoch_end
      ~CorrectSegNet.on_train_epoch_start
      ~CorrectSegNet.on_train_start
      ~CorrectSegNet.on_val_dataloader
      ~CorrectSegNet.on_validation_batch_end
      ~CorrectSegNet.on_validation_batch_start
      ~CorrectSegNet.on_validation_end
      ~CorrectSegNet.on_validation_epoch_end
      ~CorrectSegNet.on_validation_epoch_start
      ~CorrectSegNet.on_validation_model_eval
      ~CorrectSegNet.on_validation_model_train
      ~CorrectSegNet.on_validation_start
      ~CorrectSegNet.optimizer_step
      ~CorrectSegNet.optimizer_zero_grad
      ~CorrectSegNet.optimizers
      ~CorrectSegNet.parameters
      ~CorrectSegNet.predict_dataloader
      ~CorrectSegNet.predict_step
      ~CorrectSegNet.prepare_data
      ~CorrectSegNet.print
      ~CorrectSegNet.register_backward_hook
      ~CorrectSegNet.register_buffer
      ~CorrectSegNet.register_forward_hook
      ~CorrectSegNet.register_forward_pre_hook
      ~CorrectSegNet.register_full_backward_hook
      ~CorrectSegNet.register_load_state_dict_post_hook
      ~CorrectSegNet.register_module
      ~CorrectSegNet.register_parameter
      ~CorrectSegNet.requires_grad_
      ~CorrectSegNet.save_hyperparameters
      ~CorrectSegNet.set_extra_state
      ~CorrectSegNet.setup
      ~CorrectSegNet.share_memory
      ~CorrectSegNet.state_dict
      ~CorrectSegNet.summarize
      ~CorrectSegNet.tbptt_split_batch
      ~CorrectSegNet.teardown
      ~CorrectSegNet.test_dataloader
      ~CorrectSegNet.test_epoch_end
      ~CorrectSegNet.test_step
      ~CorrectSegNet.test_step_end
      ~CorrectSegNet.to
      ~CorrectSegNet.to_empty
      ~CorrectSegNet.to_onnx
      ~CorrectSegNet.to_torchscript
      ~CorrectSegNet.toggle_optimizer
      ~CorrectSegNet.train
      ~CorrectSegNet.train_dataloader
      ~CorrectSegNet.training_epoch_end
      ~CorrectSegNet.training_step
      ~CorrectSegNet.training_step_end
      ~CorrectSegNet.transfer_batch_to_device
      ~CorrectSegNet.type
      ~CorrectSegNet.unfreeze
      ~CorrectSegNet.untoggle_optimizer
      ~CorrectSegNet.val_dataloader
      ~CorrectSegNet.validation_epoch_end
      ~CorrectSegNet.validation_step
      ~CorrectSegNet.validation_step_end
      ~CorrectSegNet.xpu
      ~CorrectSegNet.zero_grad
   
   

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~CorrectSegNet.CHECKPOINT_HYPER_PARAMS_KEY
      ~CorrectSegNet.CHECKPOINT_HYPER_PARAMS_NAME
      ~CorrectSegNet.CHECKPOINT_HYPER_PARAMS_TYPE
      ~CorrectSegNet.T_destination
      ~CorrectSegNet.automatic_optimization
      ~CorrectSegNet.current_epoch
      ~CorrectSegNet.device
      ~CorrectSegNet.dtype
      ~CorrectSegNet.dump_patches
      ~CorrectSegNet.example_input_array
      ~CorrectSegNet.global_rank
      ~CorrectSegNet.global_step
      ~CorrectSegNet.hparams
      ~CorrectSegNet.hparams_initial
      ~CorrectSegNet.local_rank
      ~CorrectSegNet.logger
      ~CorrectSegNet.loggers
      ~CorrectSegNet.model_size
      ~CorrectSegNet.on_gpu
      ~CorrectSegNet.truncated_bptt_steps
      ~CorrectSegNet.use_amp
      ~CorrectSegNet.training
   
   