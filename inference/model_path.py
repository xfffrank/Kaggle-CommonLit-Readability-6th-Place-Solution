class ModelPath:
    '''
    This is the class for locating the transformer models, change 'models_dir' to the directory of the models
    '''
    models_dir = '../../models'
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


    gpr_path = f'{models_dir}/gpr_rbf/best_9_all_y_gpr'
    emb_path = f'{models_dir}/gpr_rbf/best_9_all_y_embeddings.npy'
    y_path = f'{models_dir}/gpr_rbf/best_9_all_y_y_new.npy'