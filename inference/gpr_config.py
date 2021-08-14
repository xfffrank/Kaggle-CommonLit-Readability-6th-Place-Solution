class GPR_CFG:
    '''
        This is the GPR Config class for saving/loading the .csv files, transformer models, GPR model,
        and other misc files required for training/inference.
        Each variable with comment (MUST EXIST) means the following file/folder needs to be exists before running
        any GPR_training.py or GPR_inference.py.
    '''
    models_dir = '../../models' # the directory of all the models checkpoint (MUST EXIST)
    COMPUTE_CV = False # Set to True if running inference on train.csv, set to False if running inference on test.csv
    train_csv = '../../commonlitreadabilityprize/train.csv' # original train.csv (MUST EXIST)
    test_csv = '../../commonlitreadabilityprize/test.csv' # original test.csv (MUST EXIST)
    output_csv = '../../commonlitreadabilityprize/final.csv' # the path to store the generated output file with readability predictions after inference

    gpr_path = f'{models_dir}/gpr_rbf/best_9_all_y_gpr' # the path to store or load the trained GPR model on the 9 concatenated embeddings
    emb_path = f'{models_dir}/gpr_rbf/best_9_all_y_embeddings.npy' # the path to store or load the 9 concatenated embeddings
    y_path = f'{models_dir}/gpr_rbf/best_9_all_y_y_new.npy' # the path to store or load the target label

    seed = 42  # for creating kfolds

    # The following are for loading each model checkpoint from the 'models' directory. (MUST EXIST)
    roberta_large_model= [f'{models_dir}/exp_180/exp_180_*/roberta-large_seed42_fold0.ckpt',
                        f'{models_dir}/exp_180/exp_180_*/roberta-large_seed42_fold1.ckpt',
                        f'{models_dir}/exp_180/exp_180_*/roberta-large_seed42_fold2.ckpt',
                        f'{models_dir}/exp_180/exp_180_*/roberta-large_seed42_fold3.ckpt',
                        f'{models_dir}/exp_180/exp_180_*/roberta-large_seed42_fold4.ckpt',
                        ]

    deberta_large_model = [f'{models_dir}/exp_187/exp_187_*/microsoft/deberta-large_seed42_fold0.ckpt',
                        f'{models_dir}/exp_187/exp_187_*/microsoft/deberta-large_seed42_fold1.ckpt',
                        f'{models_dir}/exp_187/exp_187_*/microsoft/deberta-large_seed42_fold2.ckpt',
                        f'{models_dir}/exp_187/exp_187_*/microsoft/deberta-large_seed42_fold3.ckpt',
                        f'{models_dir}/exp_187/exp_187_*/microsoft/deberta-large_seed42_fold4.ckpt',
                        ]

    xlnet_large_model = [f'{models_dir}/exp_284_/exp_284/xlnet-large-cased_seed42_fold0.ckpt',
                        f'{models_dir}/exp_284_/exp_284/xlnet-large-cased_seed42_fold1.ckpt',
                        f'{models_dir}/exp_284_/exp_284/xlnet-large-cased_seed42_fold2.ckpt',
                        f'{models_dir}/exp_284_/exp_284/xlnet-large-cased_seed42_fold3.ckpt',
                        f'{models_dir}/exp_284_/exp_284/xlnet-large-cased_seed42_fold4.ckpt',
                        ]

    roberta_large_squad2_model = [f'{models_dir}/exp_312_rl_squad/exp_312_*/roberta-large_seed42_fold0.ckpt',
                                f'{models_dir}/exp_312_rl_squad/exp_312_*/roberta-large_seed42_fold1.ckpt',
                                f'{models_dir}/exp_312_rl_squad/exp_312_*/roberta-large_seed42_fold2.ckpt',
                                f'{models_dir}/exp_312_rl_squad/exp_312_*/roberta-large_seed42_fold3.ckpt',
                                f'{models_dir}/exp_312_rl_squad/exp_312_*/roberta-large_seed42_fold4.ckpt',
                        ]

    longformer_large_tqa_model = [f'{models_dir}/exp_426_ll_tqa/exp_426_*/allenai/longformer-large-4096-finetuned-triviaqa_seed42_fold0.ckpt',
                        f'{models_dir}/exp_426_ll_tqa/exp_426_*/allenai/longformer-large-4096-finetuned-triviaqa_seed42_fold1.ckpt',
                        f'{models_dir}/exp_426_ll_tqa/exp_426_*/allenai/longformer-large-4096-finetuned-triviaqa_seed42_fold2.ckpt',
                        f'{models_dir}/exp_426_ll_tqa/exp_426_*/allenai/longformer-large-4096-finetuned-triviaqa_seed42_fold3.ckpt',
                        f'{models_dir}/exp_426_ll_tqa/exp_426_*/allenai/longformer-large-4096-finetuned-triviaqa_seed42_fold4.ckpt',
                        ]

    roberta_large_squad2_mlm_model = [f'{models_dir}/exp_723_rl_sq2_mlm/exp_723_*/roberta-large_seed42_fold0.ckpt',
                        f'{models_dir}/exp_723_rl_sq2_mlm/exp_723_*/roberta-large_seed42_fold1.ckpt',
                        f'{models_dir}/exp_723_rl_sq2_mlm/exp_723_*/roberta-large_seed42_fold2.ckpt',
                        f'{models_dir}/exp_723_rl_sq2_mlm/exp_723_*/roberta-large_seed42_fold3.ckpt',
                        f'{models_dir}/exp_723_rl_sq2_mlm/exp_723_*/roberta-large_seed42_fold4.ckpt',
                        ]

    bart_large_squad1_mlm_rmdp_model = [f'{models_dir}/exp_799_bl_sq1_mlm_rm_dp/exp_799_*/valhalla/bart-large-finetuned-squadv1_seed42_fold0.ckpt',
                        f'{models_dir}/exp_799_bl_sq1_mlm_rm_dp/exp_799_*/valhalla/bart-large-finetuned-squadv1_seed42_fold1.ckpt',
                        f'{models_dir}/exp_799_bl_sq1_mlm_rm_dp/exp_799_*/valhalla/bart-large-finetuned-squadv1_seed42_fold2.ckpt',
                        f'{models_dir}/exp_799_bl_sq1_mlm_rm_dp/exp_799_*/valhalla/bart-large-finetuned-squadv1_seed42_fold3.ckpt',
                        f'{models_dir}/exp_799_bl_sq1_mlm_rm_dp/exp_799_*/valhalla/bart-large-finetuned-squadv1_seed42_fold4.ckpt',
                        ]

    deberta_large_mnli_mlm_rmdp_model = [f'{models_dir}/exp_804_dl_mnli_mlm_rm_dp/exp_804_*/microsoft/deberta-large-mnli_seed42_fold0.ckpt',
                        f'{models_dir}/exp_804_dl_mnli_mlm_rm_dp/exp_804_*/microsoft/deberta-large-mnli_seed42_fold1.ckpt',
                        f'{models_dir}/exp_804_dl_mnli_mlm_rm_dp/exp_804_*/microsoft/deberta-large-mnli_seed42_fold2.ckpt',
                        f'{models_dir}/exp_804_dl_mnli_mlm_rm_dp/exp_804_*/microsoft/deberta-large-mnli_seed42_fold3.ckpt',
                        f'{models_dir}/exp_804_dl_mnli_mlm_rm_dp/exp_804_*/microsoft/deberta-large-mnli_seed42_fold4.ckpt',
                        ]

    electra_large_squad2_rmdp_model = [f'{models_dir}/exp_827_el_sq2_rm_dp/exp_827_*/ahotrod/electra_large_discriminator_squad2_512_seed42_fold0.ckpt',
                        f'{models_dir}/exp_827_el_sq2_rm_dp/exp_827_*/ahotrod/electra_large_discriminator_squad2_512_seed42_fold1.ckpt',
                        f'{models_dir}/exp_827_el_sq2_rm_dp/exp_827_*/ahotrod/electra_large_discriminator_squad2_512_seed42_fold2.ckpt',
                        f'{models_dir}/exp_827_el_sq2_rm_dp/exp_827_*/ahotrod/electra_large_discriminator_squad2_512_seed42_fold3.ckpt',
                        f'{models_dir}/exp_827_el_sq2_rm_dp/exp_827_*/ahotrod/electra_large_discriminator_squad2_512_seed42_fold4.ckpt',
                        ]


