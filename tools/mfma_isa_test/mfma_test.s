	.amdgcn_target "amdgcn-amd-amdhsa-unknown-gfx950"
	.amdhsa_code_object_version 4
	.text
	.protected	mfma_agpr_test                   ; -- Begin function mfma_agpr_test
	.globl	mfma_agpr_test
	.p2align	8
	.type	mfma_agpr_test,@function
mfma_agpr_test:                          ; @mfma_agpr_test
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x04, 0x30, 0x36, 0xe9, 0x02 ; CFA is 0 in private_wave aspace
	.cfi_undefined 16
	s_load_dwordx2 s[0:1], s[0:1], 0x0
	; A = B = all bf16 1.0 pairs
	v_mov_b32_e32 v16, 0x3f803f80
	v_mov_b32_e32 v17, 0x3f803f80
	v_mov_b32_e32 v18, 0x3f803f80
	v_mov_b32_e32 v19, 0x3f803f80
	v_mov_b32_e32 v20, 0x3f803f80
	v_mov_b32_e32 v21, 0x3f803f80
	v_mov_b32_e32 v22, 0x3f803f80
	v_mov_b32_e32 v23, 0x3f803f80
	; v_pk_mul_f32 operands: (1.0, 2.0) * (3.0, 1.5) -> (3.0, 3.0)
	v_mov_b32_e32 v4, 0x3f800000
	v_mov_b32_e32 v5, 0x40000000
	v_mov_b32_e32 v6, 0x40400000
	v_mov_b32_e32 v7, 0x3fc00000
	s_waitcnt lgkmcnt(0)
	; mfma #1: C = 0 -> every accumulator element = 16.0
	v_mfma_f32_32x32x16_bf16 a[0:15], v[16:19], v[20:23], 0
	s_nop 0
	s_nop 0
	; mfma #2: C = a[0:15] (AGPR chain) -> every element = 32.0
	v_mfma_f32_32x32x16_bf16 a[0:15], v[16:19], v[20:23], a[0:15]
	s_nop 0
	s_nop 0
	; accvgpr write/read roundtrip through a[8]
	v_mov_b32_e32 v9, 0x42c80000
	v_accvgpr_write_b32 a8, v9
	v_accvgpr_read_b32 v10, a8
	; packed f32 multiply
	v_pk_mul_f32 v[8:9], v[4:5], v[6:7]
	; read back two accumulator dwords
	v_accvgpr_read_b32 v0, a0
	v_accvgpr_read_b32 v1, a1
	; lane-relative output slot: 4 dwords at out[lane*4]
	v_mbcnt_lo_u32_b32 v11, -1, 0
	v_mbcnt_hi_u32_b32 v11, -1, v11
	v_lshlrev_b32_e32 v12, 4, v11
	; store [a0, a1, accvgpr_roundtrip, pk_mul_lo] at out[lane*4 + 0..3]
	v_lshlrev_b32_e32 v13, 2, v11
	global_store_dword v13, v0, s[0:1]
	v_add_u32_e32 v14, 4, v13
	global_store_dword v14, v1, s[0:1]
	v_add_u32_e32 v15, 8, v13
	global_store_dword v15, v10, s[0:1]
	v_add_u32_e32 v16, 12, v13
	global_store_dword v16, v8, s[0:1]
	s_endpgm
.Lfunc_end0:
	.size	mfma_agpr_test, .Lfunc_end0-mfma_agpr_test
	.cfi_endproc
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel mfma_agpr_test
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 8
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 24
		.amdhsa_next_free_sgpr 8
		.amdhsa_accum_offset 24
		.amdhsa_reserve_vcc 0
		.amdhsa_reserve_xnack_mask 1
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
                                        ; -- End function
	.set .Lmfma_agpr_test.num_vgpr, 2
	.set .Lmfma_agpr_test.num_agpr, 0
	.set .Lmfma_agpr_test.numbered_sgpr, 34
	.set .Lmfma_agpr_test.num_named_barrier, 0
	.set .Lmfma_agpr_test.private_seg_size, 0
	.set .Lmfma_agpr_test.uses_vcc, 0
	.set .Lmfma_agpr_test.uses_flat_scratch, 0
	.set .Lmfma_agpr_test.has_dyn_sized_stack, 0
	.set .Lmfma_agpr_test.has_recursion, 0
	.set .Lmfma_agpr_test.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 36
; TotalNumSgprs: 40
; NumVgprs: 24
; NumAgprs: 0
; TotalNumVgprs: 2
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 4
; VGPRBlocks: 0
; NumSGPRsForWavesPerEU: 40
; NumVGPRsForWavesPerEU: 2
; AccumOffset: 4
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 0
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.protected	__clang_ocl_kern_imp_mfma_agpr_test ; -- Begin function __clang_ocl_kern_imp_mfma_agpr_test
	.globl	__clang_ocl_kern_imp_mfma_agpr_test
	.p2align	6
	.type	__clang_ocl_kern_imp_mfma_agpr_test,@function
__clang_ocl_kern_imp_mfma_agpr_test:             ; @__clang_ocl_kern_imp_mfma_agpr_test
	.cfi_startproc
; %bb.0:
	.cfi_escape 0x0f, 0x09, 0x90, 0x40, 0x94, 0x04, 0x36, 0x24, 0x36, 0xe9, 0x02 ; 
	.cfi_llvm_register_pair 16, 62, 32, 63, 32
	.cfi_undefined 2562
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	s_mov_b32 s0, s33
	.cfi_register 65, 32
	s_mov_b32 s33, s32
	.cfi_escape 0x0f, 0x09, 0x90, 0x41, 0x94, 0x04, 0x36, 0x24, 0x36, 0xe9, 0x02 ; 
	v_mov_b32_e32 v2, 1.0
	global_store_dword v[0:1], v2, off
	.cfi_escape 0x0f, 0x09, 0x90, 0x40, 0x94, 0x04, 0x36, 0x24, 0x36, 0xe9, 0x02 ; 
	s_mov_b32 s33, s0
	s_waitcnt vmcnt(0)
	s_setpc_b64 s[30:31]
.Lfunc_end1:
	.size	__clang_ocl_kern_imp_mfma_agpr_test, .Lfunc_end1-__clang_ocl_kern_imp_mfma_agpr_test
	.cfi_endproc
                                        ; -- End function
	.set .L__clang_ocl_kern_imp_mfma_agpr_test.num_vgpr, 3
	.set .L__clang_ocl_kern_imp_mfma_agpr_test.num_agpr, 0
	.set .L__clang_ocl_kern_imp_mfma_agpr_test.numbered_sgpr, 34
	.set .L__clang_ocl_kern_imp_mfma_agpr_test.num_named_barrier, 0
	.set .L__clang_ocl_kern_imp_mfma_agpr_test.private_seg_size, 0
	.set .L__clang_ocl_kern_imp_mfma_agpr_test.uses_vcc, 0
	.set .L__clang_ocl_kern_imp_mfma_agpr_test.uses_flat_scratch, 0
	.set .L__clang_ocl_kern_imp_mfma_agpr_test.has_dyn_sized_stack, 0
	.set .L__clang_ocl_kern_imp_mfma_agpr_test.has_recursion, 0
	.set .L__clang_ocl_kern_imp_mfma_agpr_test.has_indirect_call, 0
	.section	.AMDGPU.csdata,"",@progbits
; Function info:
; codeLenInByte = 36
; TotalNumSgprs: 40
; NumVgprs: 3
; NumAgprs: 0
; TotalNumVgprs: 3
; ScratchSize: 0
; MemoryBound: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.section	.AMDGPU.gpr_maximums,"",@progbits
	.set amdgpu.max_num_vgpr, 3
	.set amdgpu.max_num_agpr, 0
	.set amdgpu.max_num_sgpr, 34
	.set amdgpu.max_num_named_barrier, 0
	.text
	.ident	"clang version 24.0.0git (/home/zhaosiying/amdgpu-sim/projects/llvm-project/clang 73f2a21fe16b34e35fd0e149564b8664e59da392)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     0
    .args:
      - .address_space:  global
        .offset:         0
        .size:           8
        .type_name:      'float*'
        .value_kind:     global_buffer
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 8
    .language:       OpenCL C
    .language_version:
      - 1
      - 2
    .max_flat_workgroup_size: 256
    .name:           mfma_agpr_test
    .private_segment_fixed_size: 0
    .sgpr_count:     40
    .sgpr_spill_count: 0
    .symbol:         mfma_agpr_test.kd
    .vgpr_count:     2
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa-unknown-gfx950
amdhsa.version:
  - 1
  - 1
...

	.end_amdgpu_metadata
