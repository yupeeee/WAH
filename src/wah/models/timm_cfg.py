__all__ = [
    "_timm_need_img_size",
]

_timm_need_img_size = [
    "bat_resnext26ts",
    "beit_base_patch16_224",
    "beit_base_patch16_384",
    "beit_large_patch16_224",
    "beit_large_patch16_384",
    "beit_large_patch16_512",
    "beitv2_base_patch16_224",
    "beitv2_large_patch16_224",
    "botnet26t_256",
    "botnet50ts_256",
    "caformer_b36",
    "caformer_m36",
    "caformer_s18",
    "caformer_s36",
    "cait_m36_384",
    "cait_m48_448",
    "cait_s24_224",
    "cait_s24_384",
    "cait_s36_384",
    "cait_xs24_384",
    "cait_xxs24_224",
    "cait_xxs24_384",
    "cait_xxs36_224",
    "cait_xxs36_384",
    "coat_lite_medium",
    "coat_lite_medium_384",
    "coat_lite_mini",
    "coat_lite_small",
    "coat_lite_tiny",
    "coat_mini",
    "coat_small",
    "coat_tiny",
    "coatnet_0_224",
    "coatnet_0_rw_224",
    "coatnet_1_224",
    "coatnet_1_rw_224",
    "coatnet_2_224",
    "coatnet_2_rw_224",
    "coatnet_3_224",
    "coatnet_3_rw_224",
    "coatnet_4_224",
    "coatnet_5_224",
    "coatnet_bn_0_rw_224",
    "coatnet_nano_cc_224",
    "coatnet_nano_rw_224",
    "coatnet_pico_rw_224",
    "coatnet_rmlp_0_rw_224",
    "coatnet_rmlp_1_rw2_224",
    "coatnet_rmlp_1_rw_224",
    "coatnet_rmlp_2_rw_224",
    "coatnet_rmlp_2_rw_384",
    "coatnet_rmlp_3_rw_224",
    "coatnet_rmlp_nano_rw_224",
    "coatnext_nano_rw_224",
    "convformer_b36",
    "convformer_m36",
    "convformer_s18",
    "convformer_s36",
    "convit_base",
    "convit_small",
    "convit_tiny",
    "convmixer_768_32",
    "convmixer_1024_20_ks9_p14",
    "convmixer_1536_20",
    "crossvit_9_240",
    "crossvit_9_dagger_240",
    "crossvit_15_240",
    "crossvit_15_dagger_240",
    "crossvit_15_dagger_408",
    "crossvit_18_240",
    "crossvit_18_dagger_240",
    "crossvit_18_dagger_408",
    "crossvit_base_240",
    "crossvit_small_240",
    "crossvit_tiny_240",
    "deit3_base_patch16_224",
    "deit3_base_patch16_384",
    "deit3_huge_patch14_224",
    "deit3_large_patch16_224",
    "deit3_large_patch16_384",
    "deit3_medium_patch16_224",
    "deit3_small_patch16_224",
    "deit3_small_patch16_384",
    "deit_base_distilled_patch16_224",
    "deit_base_distilled_patch16_384",
    "deit_base_patch16_224",
    "deit_base_patch16_384",
    "deit_small_distilled_patch16_224",
    "deit_small_patch16_224",
    "deit_tiny_distilled_patch16_224",
    "deit_tiny_patch16_224",
    "eca_botnext26ts_256",
    "eca_halonext26ts",
    "eca_resnet33ts",
    "eca_resnext26ts",
    "eca_vovnet39b",
    "efficientformer_l1",
    "efficientformer_l3",
    "efficientformer_l7",
    "efficientformerv2_l",
    "efficientformerv2_s0",
    "efficientformerv2_s1",
    "efficientformerv2_s2",
    "efficientvit_m0",
    "efficientvit_m1",
    "efficientvit_m2",
    "efficientvit_m3",
    "efficientvit_m4",
    "efficientvit_m5",
    "ese_vovnet19b_dw",
    "ese_vovnet19b_slim",
    "ese_vovnet19b_slim_dw",
    "ese_vovnet39b",
    "ese_vovnet39b_evos",
    "ese_vovnet57b",
    "ese_vovnet99b",
    "eva02_base_patch14_224",
    "eva02_base_patch14_448",
    "eva02_base_patch16_clip_224",
    "eva02_enormous_patch14_clip_224",
    "eva02_large_patch14_224",
    "eva02_large_patch14_448",
    "eva02_large_patch14_clip_224",
    "eva02_large_patch14_clip_336",
    "eva02_small_patch14_224",
    "eva02_small_patch14_336",
    "eva02_tiny_patch14_224",
    "eva02_tiny_patch14_336",
    "eva_giant_patch14_224",
    "eva_giant_patch14_336",
    "eva_giant_patch14_560",
    "eva_giant_patch14_clip_224",
    "eva_large_patch14_196",
    "eva_large_patch14_336",
    "flexivit_base",
    "flexivit_large",
    "flexivit_small",
    "gcresnet33ts",
    "gcresnet50t",
    "gcresnext26ts",
    "gcresnext50ts",
    "gcvit_base",
    "gcvit_small",
    "gcvit_tiny",
    "gcvit_xtiny",
    "gcvit_xxtiny",
    "gernet_l",
    "gernet_m",
    "gernet_s",
    "gmixer_12_224",
    "gmixer_24_224",
    "gmlp_b16_224",
    "gmlp_s16_224",
    "gmlp_ti16_224",
    "halo2botnet50ts_256",
    "halonet26t",
    "halonet50ts",
    "halonet_h1",
    "haloregnetz_b",
    "hgnet_base",
    "hgnet_small",
    "hgnet_tiny",
    "hgnetv2_b0",
    "hgnetv2_b1",
    "hgnetv2_b2",
    "hgnetv2_b3",
    "hgnetv2_b4",
    "hgnetv2_b5",
    "hgnetv2_b6",
    "hrnet_w18",
    "hrnet_w18_small",
    "hrnet_w18_small_v2",
    "hrnet_w18_ssld",
    "hrnet_w30",
    "hrnet_w32",
    "hrnet_w40",
    "hrnet_w44",
    "hrnet_w48",
    "hrnet_w48_ssld",
    "hrnet_w64",
    "lambda_resnet26rpt_256",
    "lambda_resnet26t",
    "lambda_resnet50ts",
    "lamhalobotnet50ts_256",
    "levit_128",
    "levit_128s",
    "levit_192",
    "levit_256",
    "levit_256d",
    "levit_384",
    "levit_384_s8",
    "levit_512",
    "levit_512_s8",
    "levit_512d",
    "levit_conv_128",
    "levit_conv_128s",
    "levit_conv_192",
    "levit_conv_256",
    "levit_conv_256d",
    "levit_conv_384",
    "levit_conv_384_s8",
    "levit_conv_512",
    "levit_conv_512_s8",
    "levit_conv_512d",
    "maxvit_base_tf_224",
    "maxvit_base_tf_384",
    "maxvit_base_tf_512",
    "maxvit_large_tf_224",
    "maxvit_large_tf_384",
    "maxvit_large_tf_512",
    "maxvit_nano_rw_256",
    "maxvit_pico_rw_256",
    "maxvit_rmlp_base_rw_224",
    "maxvit_rmlp_base_rw_384",
    "maxvit_rmlp_nano_rw_256",
    "maxvit_rmlp_pico_rw_256",
    "maxvit_rmlp_small_rw_224",
    "maxvit_rmlp_small_rw_256",
    "maxvit_rmlp_tiny_rw_256",
    "maxvit_small_tf_224",
    "maxvit_small_tf_384",
    "maxvit_small_tf_512",
    "maxvit_tiny_pm_256",
    "maxvit_tiny_rw_224",
    "maxvit_tiny_rw_256",
    "maxvit_tiny_tf_224",
    "maxvit_tiny_tf_384",
    "maxvit_tiny_tf_512",
    "maxvit_xlarge_tf_224",
    "maxvit_xlarge_tf_384",
    "maxvit_xlarge_tf_512",
    "maxxvit_rmlp_nano_rw_256",
    "maxxvit_rmlp_small_rw_256",
    "maxxvit_rmlp_tiny_rw_256",
    "maxxvitv2_nano_rw_256",
    "maxxvitv2_rmlp_base_rw_224",
    "maxxvitv2_rmlp_base_rw_384",
    "maxxvitv2_rmlp_large_rw_224",
    "mixer_b16_224",
    "mixer_b32_224",
    "mixer_l16_224",
    "mixer_l32_224",
    "mixer_s16_224",
    "mixer_s32_224",
    "mobileone_s0",
    "mobileone_s1",
    "mobileone_s2",
    "mobileone_s3",
    "mobileone_s4",
    "mobilevit_s",
    "mobilevit_xs",
    "mobilevit_xxs",
    "mobilevitv2_050",
    "mobilevitv2_075",
    "mobilevitv2_100",
    "mobilevitv2_125",
    "mobilevitv2_150",
    "mobilevitv2_175",
    "mobilevitv2_200",
    "mvitv2_base",
    "mvitv2_base_cls",
    "mvitv2_huge_cls",
    "mvitv2_large",
    "mvitv2_large_cls",
    "mvitv2_small",
    "mvitv2_small_cls",
    "mvitv2_tiny",
    "nest_base",
    "nest_base_jx",
    "nest_small",
    "nest_small_jx",
    "nest_tiny",
    "nest_tiny_jx",
    "pit_b_224",
    "pit_b_distilled_224",
    "pit_s_224",
    "pit_s_distilled_224",
    "pit_ti_224",
    "pit_ti_distilled_224",
    "pit_xs_224",
    "pit_xs_distilled_224",
    "poolformer_m36",
    "poolformer_m48",
    "poolformer_s12",
    "poolformer_s24",
    "poolformer_s36",
    "poolformerv2_m36",
    "poolformerv2_m48",
    "poolformerv2_s12",
    "poolformerv2_s24",
    "poolformerv2_s36",
    "regnetz_b16",
    "regnetz_b16_evos",
    "regnetz_c16",
    "regnetz_c16_evos",
    "regnetz_d8",
    "regnetz_d8_evos",
    "regnetz_d32",
    "regnetz_e8",
    "repvgg_a0",
    "repvgg_a1",
    "repvgg_a2",
    "repvgg_b0",
    "repvgg_b1",
    "repvgg_b1g4",
    "repvgg_b2",
    "repvgg_b2g4",
    "repvgg_b3",
    "repvgg_b3g4",
    "repvgg_d2se",
    "repvit_m0_9",
    "repvit_m1",
    "repvit_m1_0",
    "repvit_m1_1",
    "repvit_m1_5",
    "repvit_m2",
    "repvit_m2_3",
    "repvit_m3",
    "resmlp_12_224",
    "resmlp_24_224",
    "resmlp_36_224",
    "resmlp_big_24_224",
    "resnet32ts",
    "resnet33ts",
    "resnet51q",
    "resnet61q",
    "resnext26ts",
    "samvit_base_patch16",
    "samvit_base_patch16_224",
    "samvit_huge_patch16",
    "samvit_large_patch16",
    "sebotnet33ts_256",
    "sehalonet33ts",
    "sequencer2d_l",
    "sequencer2d_m",
    "sequencer2d_s",
    "seresnet33ts",
    "seresnext26ts",
    "swin_base_patch4_window7_224",
    "swin_base_patch4_window12_384",
    "swin_large_patch4_window7_224",
    "swin_large_patch4_window12_384",
    "swin_s3_base_224",
    "swin_s3_small_224",
    "swin_s3_tiny_224",
    "swin_small_patch4_window7_224",
    "swin_tiny_patch4_window7_224",
    "swinv2_base_window8_256",
    "swinv2_base_window12_192",
    "swinv2_base_window12to16_192to256",
    "swinv2_base_window12to24_192to384",
    "swinv2_base_window16_256",
    "swinv2_cr_base_224",
    "swinv2_cr_base_384",
    "swinv2_cr_base_ns_224",
    "swinv2_cr_giant_224",
    "swinv2_cr_giant_384",
    "swinv2_cr_huge_224",
    "swinv2_cr_huge_384",
    "swinv2_cr_large_224",
    "swinv2_cr_large_384",
    "swinv2_cr_small_224",
    "swinv2_cr_small_384",
    "swinv2_cr_small_ns_224",
    "swinv2_cr_small_ns_256",
    "swinv2_cr_tiny_224",
    "swinv2_cr_tiny_384",
    "swinv2_cr_tiny_ns_224",
    "swinv2_large_window12_192",
    "swinv2_large_window12to16_192to256",
    "swinv2_large_window12to24_192to384",
    "swinv2_small_window8_256",
    "swinv2_small_window16_256",
    "swinv2_tiny_window8_256",
    "swinv2_tiny_window16_256",
    "tnt_b_patch16_224",
    "tnt_s_patch16_224",
    "twins_pcpvt_base",
    "twins_pcpvt_large",
    "twins_pcpvt_small",
    "twins_svt_base",
    "twins_svt_large",
    "twins_svt_small",
    "visformer_small",
    "visformer_tiny",
    "vit_base_patch8_224",
    "vit_base_patch14_dinov2",
    "vit_base_patch14_reg4_dinov2",
    "vit_base_patch16_18x2_224",
    "vit_base_patch16_224",
    "vit_base_patch16_224_miil",
    "vit_base_patch16_384",
    "vit_base_patch16_clip_224",
    "vit_base_patch16_clip_384",
    "vit_base_patch16_clip_quickgelu_224",
    "vit_base_patch16_gap_224",
    "vit_base_patch16_plus_240",
    "vit_base_patch16_reg4_gap_256",
    "vit_base_patch16_rpn_224",
    "vit_base_patch16_siglip_224",
    "vit_base_patch16_siglip_256",
    "vit_base_patch16_siglip_384",
    "vit_base_patch16_siglip_512",
    "vit_base_patch16_xp_224",
    "vit_base_patch32_224",
    "vit_base_patch32_384",
    "vit_base_patch32_clip_224",
    "vit_base_patch32_clip_256",
    "vit_base_patch32_clip_384",
    "vit_base_patch32_clip_448",
    "vit_base_patch32_clip_quickgelu_224",
    "vit_base_patch32_plus_256",
    "vit_base_r26_s32_224",
    "vit_base_r50_s16_224",
    "vit_base_r50_s16_384",
    "vit_base_resnet26d_224",
    "vit_base_resnet50d_224",
    "vit_giant_patch14_224",
    "vit_giant_patch14_clip_224",
    "vit_giant_patch14_dinov2",
    "vit_giant_patch14_reg4_dinov2",
    "vit_giant_patch16_gap_224",
    "vit_gigantic_patch14_224",
    "vit_gigantic_patch14_clip_224",
    "vit_huge_patch14_224",
    "vit_huge_patch14_clip_224",
    "vit_huge_patch14_clip_336",
    "vit_huge_patch14_clip_378",
    "vit_huge_patch14_clip_quickgelu_224",
    "vit_huge_patch14_clip_quickgelu_378",
    "vit_huge_patch14_gap_224",
    "vit_huge_patch14_xp_224",
    "vit_huge_patch16_gap_448",
    "vit_large_patch14_224",
    "vit_large_patch14_clip_224",
    "vit_large_patch14_clip_336",
    "vit_large_patch14_clip_quickgelu_224",
    "vit_large_patch14_clip_quickgelu_336",
    "vit_large_patch14_dinov2",
    "vit_large_patch14_reg4_dinov2",
    "vit_large_patch14_xp_224",
    "vit_large_patch16_224",
    "vit_large_patch16_384",
    "vit_large_patch16_siglip_256",
    "vit_large_patch16_siglip_384",
    "vit_large_patch32_224",
    "vit_large_patch32_384",
    "vit_large_r50_s32_224",
    "vit_large_r50_s32_384",
    "vit_medium_patch16_gap_240",
    "vit_medium_patch16_gap_256",
    "vit_medium_patch16_gap_384",
    "vit_medium_patch16_reg4_256",
    "vit_medium_patch16_reg4_gap_256",
    "vit_relpos_base_patch16_224",
    "vit_relpos_base_patch16_cls_224",
    "vit_relpos_base_patch16_clsgap_224",
    "vit_relpos_base_patch16_plus_240",
    "vit_relpos_base_patch16_rpn_224",
    "vit_relpos_base_patch32_plus_rpn_256",
    "vit_relpos_medium_patch16_224",
    "vit_relpos_medium_patch16_cls_224",
    "vit_relpos_medium_patch16_rpn_224",
    "vit_relpos_small_patch16_224",
    "vit_relpos_small_patch16_rpn_224",
    "vit_small_patch8_224",
    "vit_small_patch14_dinov2",
    "vit_small_patch14_reg4_dinov2",
    "vit_small_patch16_18x2_224",
    "vit_small_patch16_36x1_224",
    "vit_small_patch16_224",
    "vit_small_patch16_384",
    "vit_small_patch32_224",
    "vit_small_patch32_384",
    "vit_small_r26_s32_224",
    "vit_small_r26_s32_384",
    "vit_small_resnet26d_224",
    "vit_small_resnet50d_s16_224",
    "vit_so150m_patch16_reg4_gap_256",
    "vit_so150m_patch16_reg4_map_256",
    "vit_so400m_patch14_siglip_224",
    "vit_so400m_patch14_siglip_384",
    "vit_srelpos_medium_patch16_224",
    "vit_srelpos_small_patch16_224",
    "vit_tiny_patch16_224",
    "vit_tiny_patch16_384",
    "vit_tiny_r_s16_p8_224",
    "vit_tiny_r_s16_p8_384",
    "volo_d1_224",
    "volo_d1_384",
    "volo_d2_224",
    "volo_d2_384",
    "volo_d3_224",
    "volo_d3_448",
    "volo_d4_224",
    "volo_d4_448",
    "volo_d5_224",
    "volo_d5_448",
    "volo_d5_512",
    "vovnet39a",
    "vovnet57a",
    "xcit_large_24_p8_224",
    "xcit_large_24_p8_384",
    "xcit_large_24_p16_224",
    "xcit_large_24_p16_384",
    "xcit_medium_24_p8_224",
    "xcit_medium_24_p8_384",
    "xcit_medium_24_p16_224",
    "xcit_medium_24_p16_384",
    "xcit_nano_12_p8_224",
    "xcit_nano_12_p8_384",
    "xcit_nano_12_p16_224",
    "xcit_nano_12_p16_384",
    "xcit_small_12_p8_224",
    "xcit_small_12_p8_384",
    "xcit_small_12_p16_224",
    "xcit_small_12_p16_384",
    "xcit_small_24_p8_224",
    "xcit_small_24_p8_384",
    "xcit_small_24_p16_224",
    "xcit_small_24_p16_384",
    "xcit_tiny_12_p8_224",
    "xcit_tiny_12_p8_384",
    "xcit_tiny_12_p16_224",
    "xcit_tiny_12_p16_384",
    "xcit_tiny_24_p8_224",
    "xcit_tiny_24_p8_384",
    "xcit_tiny_24_p16_224",
    "xcit_tiny_24_p16_384",
]
