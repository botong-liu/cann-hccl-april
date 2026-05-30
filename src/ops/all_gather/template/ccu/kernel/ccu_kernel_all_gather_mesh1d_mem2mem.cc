/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_kernel_all_gather_mesh1d_mem2mem.h"
#include "ccu_kernel_alg_base.h"

namespace ops_hccl {
using namespace hcomm;

constexpr int OUTPUT_XN_ID = 1;
constexpr int TOKEN_XN_ID = 2;
constexpr int CKE_IDX_0 = 0;
constexpr int CKE_IDX_1 = 1;
constexpr uint64_t CCU_MS_SIZE = 4096;
constexpr uint64_t LOCAL_COPY_MS = 8;
constexpr int POST_SYNC_ID = 3;
constexpr uint32_t INVALID_CHANNEL_IDX = static_cast<uint32_t>(-1);

CcuKernelAllGatherMesh1DMem2Mem::CcuKernelAllGatherMesh1DMem2Mem(const CcuKernelArg &arg)
    : CcuKernelAlgBase(arg)
{
    const CcuKernelArgAllGatherMesh1DMem2Mem *kernelArg
        = dynamic_cast<const CcuKernelArgAllGatherMesh1DMem2Mem *>(&arg);
    rankId_         = kernelArg->rankId_;
    rankSize_       = kernelArg->dimSize_;
    channels_       = kernelArg->channels;
    mainChannelIdxByRank_ = kernelArg->mainChannelIdxByRank_;
    sharedChannelIdxByRank_ = kernelArg->sharedChannelIdxByRank_;

    HCCL_INFO(
        "[CcuKernelAllGatherMesh1DMem2Mem] Init, KernelArgs are rankId[%u], rankSize_[%u]",
        rankId_, rankSize_);
}

HcclResult CcuKernelAllGatherMesh1DMem2Mem::InitResource()
{
    localInput_           = CreateVariable();
    if (channels_.size() == 0) {
        HCCL_ERROR("[CcuKernelAllGatherMesh1DMem2Mem] channels is empty!");
        return HcclResult::HCCL_E_INTERNAL;
    }
    if (mainChannelIdxByRank_.empty()) {
        mainChannelIdxByRank_.assign(rankSize_, INVALID_CHANNEL_IDX);
        uint32_t channelIdx = 0;
        for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
            if (peerId != rankId_) {
                CHK_PRT_RET(channelIdx >= channels_.size(),
                    HCCL_ERROR("[CcuKernelAllGatherMesh1DMem2Mem] legacy channel layout invalid, "
                               "channelIdx[%u], channels size[%zu].", channelIdx, channels_.size()),
                    HcclResult::HCCL_E_INTERNAL);
                mainChannelIdxByRank_[peerId] = channelIdx++;
            }
        }
    }
    CHK_PRT_RET(mainChannelIdxByRank_.size() != rankSize_,
        HCCL_ERROR("[CcuKernelAllGatherMesh1DMem2Mem] invalid main channel map size[%zu], rankSize[%llu].",
                   mainChannelIdxByRank_.size(), rankSize_), HcclResult::HCCL_E_INTERNAL);
    if (sharedChannelIdxByRank_.empty()) {
        sharedChannelIdxByRank_.assign(rankSize_, INVALID_CHANNEL_IDX);
    }
    CHK_PRT_RET(sharedChannelIdxByRank_.size() != rankSize_,
        HCCL_ERROR("[CcuKernelAllGatherMesh1DMem2Mem] invalid shared channel map size[%zu], rankSize[%llu].",
                   sharedChannelIdxByRank_.size(), rankSize_), HcclResult::HCCL_E_INTERNAL);

    for (uint64_t peerId = 0; peerId < rankSize_; peerId++) {
        if (peerId == rankId_) {
            output_.push_back(CreateVariable());
            token_.push_back(CreateVariable());
            sharedOutput_.push_back(CreateVariable());
            sharedToken_.push_back(CreateVariable());
        } else {
            uint32_t mainChannelIdx = mainChannelIdxByRank_[peerId];
            CHK_PRT_RET(mainChannelIdx >= channels_.size(),
                HCCL_ERROR("[CcuKernelAllGatherMesh1DMem2Mem] invalid main channel idx[%u], peerId[%u], "
                           "channels size[%zu].", mainChannelIdx, peerId, channels_.size()), HcclResult::HCCL_E_INTERNAL);
            HCCL_DEBUG("[CcuKernelAllGatherMesh1DMem2Mem] MyRank[%u], PeerId[%u], MainChannelId[%u]",
                       rankId_, peerId, mainChannelIdx);
            CcuRep::Variable inputVar, tokenVar;
            CHK_RET(CreateVariable(channels_[mainChannelIdx], OUTPUT_XN_ID, &inputVar));
            output_.push_back(inputVar); // 获取channel中id=0的Var来传递output
            CHK_RET(CreateVariable(channels_[mainChannelIdx], TOKEN_XN_ID, &tokenVar));
            token_.push_back(tokenVar);

            uint32_t sharedChannelIdx = sharedChannelIdxByRank_[peerId];
            if (sharedChannelIdx < channels_.size()) {
                HCCL_DEBUG("[CcuKernelAllGatherMesh1DMem2Mem] MyRank[%u], PeerId[%u], SharedChannelId[%u]",
                           rankId_, peerId, sharedChannelIdx);
                CcuRep::Variable sharedOutputVar, sharedTokenVar;
                CHK_RET(CreateVariable(channels_[sharedChannelIdx], OUTPUT_XN_ID, &sharedOutputVar));
                sharedOutput_.push_back(sharedOutputVar);
                CHK_RET(CreateVariable(channels_[sharedChannelIdx], TOKEN_XN_ID, &sharedTokenVar));
                sharedToken_.push_back(sharedTokenVar);
                sharedEventMask_ |= (1 << peerId);
            } else {
                sharedOutput_.push_back(CreateVariable());
                sharedToken_.push_back(CreateVariable());
                HCCL_DEBUG("[CcuKernelAllGatherMesh1DMem2Mem] MyRank[%u], PeerId[%u] has no shared channel.",
                           rankId_, peerId);
            }
        }
    }
    currentRankSliceInputOffset_  = CreateVariable();
    currentRankSliceOutputOffset_ = CreateVariable();
    inputRepeatStride_            = CreateVariable();
    outputRepeatStride_           = CreateVariable();
    tmpRepeatNum_                 = CreateVariable();
    normalSliceSize_              = CreateVariable();
    lastSliceSize_                = CreateVariable();
    mainSliceSize_                = CreateVariable();
    sharedSliceSize_              = CreateVariable();
    constVar1_                    = CreateVariable();
    constVar1_                    = 1;
    repeatTimeflag_               = CreateVariable();
    repeatTimeflag_               = 0;
    isInputOutputEqual_           = CreateVariable();
    localGoSize_                  = CreateGroupOpSize();

    src = CreateLocalAddr();
    src_loccopy = CreateLocalAddr();
    remote_src= CreateLocalAddr();
    shared_src = CreateLocalAddr();

    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx == rankId_) {
            dst.push_back({});
            sharedDst.push_back({});
        }
        else {
            dst.push_back(CreateRemoteAddr());
            sharedDst.push_back(CreateRemoteAddr());
        }
    }

    event_ = CreateCompletedEvent();
    sharedEvent_ = CreateCompletedEvent();
    return HcclResult::HCCL_SUCCESS;
}

void CcuKernelAllGatherMesh1DMem2Mem::LoadArgs()
{
    Load(localInput_);
    Load(output_[rankId_]);
    Load(token_[rankId_]);
    Load(currentRankSliceInputOffset_);
    Load(currentRankSliceOutputOffset_);
    Load(tmpRepeatNum_);
    Load(inputRepeatStride_);
    Load(outputRepeatStride_);
    Load(normalSliceSize_);
    Load(lastSliceSize_);
    Load(mainSliceSize_);
    Load(sharedSliceSize_);
    Load(isInputOutputEqual_);
    Load(localGoSize_);
    return;
}

void CcuKernelAllGatherMesh1DMem2Mem::PreSync()
{
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, OUTPUT_XN_ID, output_[rankId_], 1 << OUTPUT_XN_ID);
        NotifyRecord(channel, CKE_IDX_0, TOKEN_XN_ID, token_[rankId_], 1 << TOKEN_XN_ID);
    }

    uint16_t allBit  = 1 << OUTPUT_XN_ID | 1 << TOKEN_XN_ID;
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, allBit);
    }
    return;
}

void CcuKernelAllGatherMesh1DMem2Mem::PostSync()
{
    for (ChannelHandle channel : channels_) {
        NotifyRecord(channel, CKE_IDX_0, 1 << POST_SYNC_ID);         // bit index = 4, 用作后同步。cke都可以用同一个，所以都是CKE_IDX_0
    }
    for (ChannelHandle channel : channels_) {
        NotifyWait(channel, CKE_IDX_0, 1 << POST_SYNC_ID);
    }
}

void CcuKernelAllGatherMesh1DMem2Mem::DoAllGather()
{
    for (uint64_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx == rankId_) {
            CCU_IF(isInputOutputEqual_ != 0)
            {
                event_.SetMask(1 << rankIdx);
                RecordEvent(event_);
            }
            CCU_IF(isInputOutputEqual_ == 0)
            {
                event_.SetMask(1 << rankIdx);
                GroupCopy(remote_src, src_loccopy, localGoSize_);
                RecordEvent(event_);
            }
        } else {
            uint32_t mainChannelIdx = mainChannelIdxByRank_[rankIdx];
            uint32_t sharedChannelIdx = sharedChannelIdxByRank_[rankIdx];
            bool hasSharedChannel = sharedChannelIdx < channels_.size();
            CCU_IF(normalSliceSize_ != 0)
            {
                event_.SetMask(1 << rankIdx);
                if (hasSharedChannel) {
                    WriteNb(channels_[mainChannelIdx], dst[rankIdx], src, mainSliceSize_, event_);
                } else {
                    WriteNb(channels_[mainChannelIdx], dst[rankIdx], src, normalSliceSize_, event_);
                }
            }
            if (hasSharedChannel) {
                CCU_IF(sharedSliceSize_ != 0)
                {
                    sharedEvent_.SetMask(1 << rankIdx);
                    WriteNb(channels_[sharedChannelIdx], sharedDst[rankIdx], shared_src, sharedSliceSize_, sharedEvent_);
                }
            }
        }
    }
    event_.SetMask((1 << rankSize_) - 1);
    WaitEvent(event_);
    if (sharedEventMask_ != 0) {
        CCU_IF(sharedSliceSize_ != 0)
        {
            sharedEvent_.SetMask(sharedEventMask_);
            WaitEvent(sharedEvent_);
        }
    }
}

void CcuKernelAllGatherMesh1DMem2Mem::DoRepeatAllGather()
{
    src.addr = localInput_;
    src.addr += currentRankSliceInputOffset_;
    src.token = token_[rankId_];

    src_loccopy.addr = localInput_;
    src_loccopy.addr += currentRankSliceInputOffset_;
    src_loccopy.token = token_[rankId_];

    shared_src.addr = localInput_;
    shared_src.addr += currentRankSliceInputOffset_;
    shared_src.addr += mainSliceSize_;
    shared_src.token = token_[rankId_];

    for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
        if (rankIdx == rankId_){
            remote_src.addr = output_[rankId_];
            remote_src.addr += currentRankSliceOutputOffset_;
            remote_src.token = token_[rankId_];
        } else {
            dst[rankIdx].addr = output_[rankIdx];
            dst[rankIdx].addr += currentRankSliceOutputOffset_;
            dst[rankIdx].token = token_[rankIdx];
            if (sharedChannelIdxByRank_[rankIdx] < channels_.size()) {
                sharedDst[rankIdx].addr = sharedOutput_[rankIdx];
                sharedDst[rankIdx].addr += currentRankSliceOutputOffset_;
                sharedDst[rankIdx].addr += mainSliceSize_;
                sharedDst[rankIdx].token = sharedToken_[rankIdx];
            }
        }
    }
    CCU_WHILE(tmpRepeatNum_ != UINT64_MAX)
    {
        tmpRepeatNum_ += constVar1_;
        CCU_IF(repeatTimeflag_ != 0)
        {
            src.addr += inputRepeatStride_;
            shared_src.addr += inputRepeatStride_;
            for (uint32_t rankIdx = 0; rankIdx < rankSize_; rankIdx++) {
                if (rankIdx == rankId_){
                    remote_src.addr += outputRepeatStride_;
                } else {
                    dst[rankIdx].addr += outputRepeatStride_;
                    if (sharedChannelIdxByRank_[rankIdx] < channels_.size()) {
                        sharedDst[rankIdx].addr += outputRepeatStride_;
                    }
                }
            }
        }
        CCU_IF(normalSliceSize_ != 0)
        {
            DoAllGather();
        }
        repeatTimeflag_ = 1;
    }
}

HcclResult CcuKernelAllGatherMesh1DMem2Mem::Algorithm()
{
    HCCL_INFO("[CcuKernelAllGatherMesh1DMem2Mem] AllgatherMesh1D run.");
    InitResource();
    LoadArgs();
    PreSync();
    DoRepeatAllGather();
    PostSync();
    HCCL_INFO("[CcuKernelAllGatherMesh1DMem2Mem] AllgatherMesh1D end.");
    return HcclResult::HCCL_SUCCESS;
}

std::vector<uint64_t> CcuKernelAllGatherMesh1DMem2Mem::GeneArgs(const CcuTaskArg &arg)
{
    const CcuTaskArgAllGatherMesh1DMem2Mem *taskArg
        = dynamic_cast<const CcuTaskArgAllGatherMesh1DMem2Mem *>(&arg);
    uint64_t inputAddr                   = taskArg->inputAddr_;
    uint64_t outputAddr                  = taskArg->outputAddr_;
    uint64_t token                       = taskArg->token_;
    
    uint64_t currentRankSliceInputOffset  = taskArg->inputSliceStride_ * rankId_;
    uint64_t currentRankSliceOutputOffset = taskArg->outputSliceStride_ * rankId_;
    uint64_t tmpRepeatNum                 = UINT64_MAX - taskArg->repeatNum_;
    uint64_t inputRepeatStride            = taskArg->inputRepeatStride_;
    uint64_t outputRepeatStride           = taskArg->outputRepeatStride_;
    uint64_t normalSliceSize              = taskArg->normalSliceSize_;
    uint64_t lastSliceSize                = taskArg->lastSliceSize_;
    uint64_t isInputOutputEqual           = taskArg->isInputOutputEqual_;
    uint64_t mainSliceSize                = taskArg->mainSliceSize_;
    uint64_t sharedSliceSize              = taskArg->sharedSliceSize_;

    auto goSize                           = CalGoSize(normalSliceSize);

    std::vector<uint64_t> taskArgs = {inputAddr,
                                      outputAddr,
                                      token,
                                      currentRankSliceInputOffset,
                                      currentRankSliceOutputOffset,
                                      tmpRepeatNum,
                                      inputRepeatStride,
                                      outputRepeatStride,
                                      normalSliceSize,
                                      lastSliceSize,
                                      mainSliceSize,
                                      sharedSliceSize,
                                      isInputOutputEqual,
                                      goSize[0],
                                      goSize[1],
                                      goSize[2],
                                      goSize[3]};

    HCCL_INFO("[CcuKernelAllGatherMesh1DMem2Mem] TaskArgs: inputAddr[%llu], outputAddr[%llu], "
        "currentRankSliceInputOffset[%llu], currentRankSliceOutputOffset[%llu], "
        "repeatNum[%llu],inputRepeatStride[%llu], outputRepeatStride[%llu], normalSliceSize[%llu], "
        "lastSliceSize[%llu], mainSliceSize[%llu], sharedSliceSize[%llu]",
        inputAddr, outputAddr, currentRankSliceInputOffset, currentRankSliceOutputOffset, tmpRepeatNum,
        inputRepeatStride, outputRepeatStride, normalSliceSize, lastSliceSize, mainSliceSize, sharedSliceSize);
    return taskArgs;
}

} // namespace ops_hccl