package com.google.mediapipe.tracking;

import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.mediapipe.tracking.MotionModelsProto;
import com.google.protobuf.AbstractMessageLite;
import com.google.protobuf.ByteString;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.ExtensionRegistryLite;
import com.google.protobuf.GeneratedMessageLite;
import com.google.protobuf.Internal;
import com.google.protobuf.InvalidProtocolBufferException;
import com.google.protobuf.MessageLiteOrBuilder;
import com.google.protobuf.Parser;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.util.Collections;
import java.util.List;
import org.opencv.calib3d.Calib3d;

/* JADX WARN: Classes with same name are omitted:
  classes2.dex
 */
/* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/TrackingProto.class */
public final class TrackingProto {

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/TrackingProto$MotionBoxInternalStateOrBuilder.class */
    public interface MotionBoxInternalStateOrBuilder extends MessageLiteOrBuilder {
        List<Float> getPosXList();

        int getPosXCount();

        float getPosX(int index);

        List<Float> getPosYList();

        int getPosYCount();

        float getPosY(int index);

        List<Float> getDxList();

        int getDxCount();

        float getDx(int index);

        List<Float> getDyList();

        int getDyCount();

        float getDy(int index);

        List<Float> getCameraDxList();

        int getCameraDxCount();

        float getCameraDx(int index);

        List<Float> getCameraDyList();

        int getCameraDyCount();

        float getCameraDy(int index);

        List<Integer> getTrackIdList();

        int getTrackIdCount();

        int getTrackId(int index);

        List<Float> getInlierScoreList();

        int getInlierScoreCount();

        float getInlierScore(int index);
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/TrackingProto$MotionBoxStateOrBuilder.class */
    public interface MotionBoxStateOrBuilder extends MessageLiteOrBuilder {
        boolean hasPosX();

        float getPosX();

        boolean hasPosY();

        float getPosY();

        boolean hasWidth();

        float getWidth();

        boolean hasHeight();

        float getHeight();

        boolean hasScale();

        float getScale();

        boolean hasRotation();

        float getRotation();

        boolean hasQuad();

        MotionBoxState.Quad getQuad();

        boolean hasAspectRatio();

        float getAspectRatio();

        boolean hasRequestGrouping();

        boolean getRequestGrouping();

        boolean hasPnpHomography();

        MotionModelsProto.Homography getPnpHomography();

        boolean hasDx();

        float getDx();

        boolean hasDy();

        float getDy();

        boolean hasKineticEnergy();

        float getKineticEnergy();

        boolean hasPriorWeight();

        float getPriorWeight();

        boolean hasTrackStatus();

        MotionBoxState.TrackStatus getTrackStatus();

        boolean hasSpatialPriorGridSize();

        int getSpatialPriorGridSize();

        List<Float> getSpatialPriorList();

        int getSpatialPriorCount();

        float getSpatialPrior(int index);

        List<Float> getSpatialConfidenceList();

        int getSpatialConfidenceCount();

        float getSpatialConfidence(int index);

        boolean hasPriorDiff();

        float getPriorDiff();

        boolean hasMotionDisparity();

        float getMotionDisparity();

        boolean hasBackgroundDiscrimination();

        float getBackgroundDiscrimination();

        boolean hasInlierCenterX();

        float getInlierCenterX();

        boolean hasInlierCenterY();

        float getInlierCenterY();

        boolean hasInlierSum();

        float getInlierSum();

        boolean hasInlierRatio();

        float getInlierRatio();

        boolean hasInlierWidth();

        float getInlierWidth();

        boolean hasInlierHeight();

        float getInlierHeight();

        List<Integer> getInlierIdsList();

        int getInlierIdsCount();

        int getInlierIds(int index);

        List<Integer> getInlierIdMatchPosList();

        int getInlierIdMatchPosCount();

        int getInlierIdMatchPos(int index);

        List<Integer> getInlierLengthList();

        int getInlierLengthCount();

        int getInlierLength(int index);

        List<Integer> getOutlierIdsList();

        int getOutlierIdsCount();

        int getOutlierIds(int index);

        List<Integer> getOutlierIdMatchPosList();

        int getOutlierIdMatchPosCount();

        int getOutlierIdMatchPos(int index);

        boolean hasTrackingConfidence();

        float getTrackingConfidence();

        boolean hasInternal();

        MotionBoxInternalState getInternal();
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/TrackingProto$TrackStepOptionsOrBuilder.class */
    public interface TrackStepOptionsOrBuilder extends MessageLiteOrBuilder {
        boolean hasTrackingDegrees();

        TrackStepOptions.TrackingDegrees getTrackingDegrees();

        boolean hasTrackObjectAndCamera();

        boolean getTrackObjectAndCamera();

        boolean hasIrlsIterations();

        int getIrlsIterations();

        boolean hasSpatialSigma();

        float getSpatialSigma();

        boolean hasMinMotionSigma();

        float getMinMotionSigma();

        boolean hasRelativeMotionSigma();

        float getRelativeMotionSigma();

        boolean hasMotionDisparityLowLevel();

        float getMotionDisparityLowLevel();

        boolean hasMotionDisparityHighLevel();

        float getMotionDisparityHighLevel();

        boolean hasDisparityDecay();

        float getDisparityDecay();

        boolean hasMotionPriorWeight();

        float getMotionPriorWeight();

        boolean hasBackgroundDiscriminationLowLevel();

        float getBackgroundDiscriminationLowLevel();

        boolean hasBackgroundDiscriminationHighLevel();

        float getBackgroundDiscriminationHighLevel();

        boolean hasInlierCenterRelativeDistance();

        float getInlierCenterRelativeDistance();

        boolean hasInlierSpringForce();

        float getInlierSpringForce();

        boolean hasKineticCenterRelativeDistance();

        float getKineticCenterRelativeDistance();

        boolean hasKineticSpringForce();

        float getKineticSpringForce();

        boolean hasKineticSpringForceMinKineticEnergy();

        float getKineticSpringForceMinKineticEnergy();

        boolean hasVelocityUpdateWeight();

        float getVelocityUpdateWeight();

        boolean hasMaxTrackFailures();

        int getMaxTrackFailures();

        boolean hasExpansionSize();

        float getExpansionSize();

        boolean hasInlierLowWeight();

        float getInlierLowWeight();

        boolean hasInlierHighWeight();

        float getInlierHighWeight();

        boolean hasKineticEnergyDecay();

        float getKineticEnergyDecay();

        boolean hasPriorWeightIncrease();

        float getPriorWeightIncrease();

        boolean hasLowKineticEnergy();

        float getLowKineticEnergy();

        boolean hasHighKineticEnergy();

        float getHighKineticEnergy();

        boolean hasReturnInternalState();

        boolean getReturnInternalState();

        boolean hasUsePostEstimationWeightsForState();

        boolean getUsePostEstimationWeightsForState();

        boolean hasComputeSpatialPrior();

        boolean getComputeSpatialPrior();

        boolean hasIrlsInitialization();

        TrackStepOptions.IrlsInitialization getIrlsInitialization();

        boolean hasStaticMotionTemporalRatio();

        float getStaticMotionTemporalRatio();

        boolean hasCancelTrackingWithOcclusionOptions();

        TrackStepOptions.CancelTrackingWithOcclusionOptions getCancelTrackingWithOcclusionOptions();

        boolean hasObjectSimilarityMinContdInliers();

        int getObjectSimilarityMinContdInliers();

        boolean hasBoxSimilarityMaxScale();

        float getBoxSimilarityMaxScale();

        boolean hasBoxSimilarityMaxRotation();

        float getBoxSimilarityMaxRotation();

        boolean hasQuadHomographyMaxScale();

        float getQuadHomographyMaxScale();

        boolean hasQuadHomographyMaxRotation();

        float getQuadHomographyMaxRotation();

        boolean hasCameraIntrinsics();

        TrackStepOptions.CameraIntrinsics getCameraIntrinsics();

        boolean hasForcedPnpTracking();

        boolean getForcedPnpTracking();
    }

    private TrackingProto() {
    }

    public static void registerAllExtensions(ExtensionRegistryLite registry) {
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/TrackingProto$MotionBoxState.class */
    public static final class MotionBoxState extends GeneratedMessageLite<MotionBoxState, Builder> implements MotionBoxStateOrBuilder {
        private int bitField0_;
        public static final int POS_X_FIELD_NUMBER = 1;
        private float posX_;
        public static final int POS_Y_FIELD_NUMBER = 2;
        private float posY_;
        public static final int WIDTH_FIELD_NUMBER = 3;
        private float width_;
        public static final int HEIGHT_FIELD_NUMBER = 4;
        private float height_;
        public static final int SCALE_FIELD_NUMBER = 5;
        public static final int ROTATION_FIELD_NUMBER = 30;
        private float rotation_;
        public static final int QUAD_FIELD_NUMBER = 34;
        private Quad quad_;
        public static final int ASPECT_RATIO_FIELD_NUMBER = 35;
        private float aspectRatio_;
        public static final int REQUEST_GROUPING_FIELD_NUMBER = 37;
        private boolean requestGrouping_;
        public static final int PNP_HOMOGRAPHY_FIELD_NUMBER = 36;
        private MotionModelsProto.Homography pnpHomography_;
        public static final int DX_FIELD_NUMBER = 7;
        private float dx_;
        public static final int DY_FIELD_NUMBER = 8;
        private float dy_;
        public static final int KINETIC_ENERGY_FIELD_NUMBER = 17;
        private float kineticEnergy_;
        public static final int PRIOR_WEIGHT_FIELD_NUMBER = 9;
        private float priorWeight_;
        public static final int TRACK_STATUS_FIELD_NUMBER = 10;
        private int trackStatus_;
        public static final int SPATIAL_PRIOR_GRID_SIZE_FIELD_NUMBER = 11;
        public static final int SPATIAL_PRIOR_FIELD_NUMBER = 12;
        public static final int SPATIAL_CONFIDENCE_FIELD_NUMBER = 13;
        public static final int PRIOR_DIFF_FIELD_NUMBER = 14;
        private float priorDiff_;
        public static final int MOTION_DISPARITY_FIELD_NUMBER = 15;
        private float motionDisparity_;
        public static final int BACKGROUND_DISCRIMINATION_FIELD_NUMBER = 16;
        private float backgroundDiscrimination_;
        public static final int INLIER_CENTER_X_FIELD_NUMBER = 18;
        private float inlierCenterX_;
        public static final int INLIER_CENTER_Y_FIELD_NUMBER = 19;
        private float inlierCenterY_;
        public static final int INLIER_SUM_FIELD_NUMBER = 24;
        private float inlierSum_;
        public static final int INLIER_RATIO_FIELD_NUMBER = 25;
        private float inlierRatio_;
        public static final int INLIER_WIDTH_FIELD_NUMBER = 22;
        private float inlierWidth_;
        public static final int INLIER_HEIGHT_FIELD_NUMBER = 23;
        private float inlierHeight_;
        public static final int INLIER_IDS_FIELD_NUMBER = 26;
        public static final int INLIER_ID_MATCH_POS_FIELD_NUMBER = 31;
        public static final int INLIER_LENGTH_FIELD_NUMBER = 27;
        public static final int OUTLIER_IDS_FIELD_NUMBER = 28;
        public static final int OUTLIER_ID_MATCH_POS_FIELD_NUMBER = 32;
        public static final int TRACKING_CONFIDENCE_FIELD_NUMBER = 33;
        private float trackingConfidence_;
        public static final int INTERNAL_FIELD_NUMBER = 29;
        private MotionBoxInternalState internal_;
        private static final MotionBoxState DEFAULT_INSTANCE;
        private static volatile Parser<MotionBoxState> PARSER;
        private int spatialPriorMemoizedSerializedSize = -1;
        private int spatialConfidenceMemoizedSerializedSize = -1;
        private int inlierIdsMemoizedSerializedSize = -1;
        private int inlierIdMatchPosMemoizedSerializedSize = -1;
        private int inlierLengthMemoizedSerializedSize = -1;
        private int outlierIdsMemoizedSerializedSize = -1;
        private int outlierIdMatchPosMemoizedSerializedSize = -1;
        private float scale_ = 1.0f;
        private int spatialPriorGridSize_ = 10;
        private Internal.FloatList spatialPrior_ = emptyFloatList();
        private Internal.FloatList spatialConfidence_ = emptyFloatList();
        private Internal.IntList inlierIds_ = emptyIntList();
        private Internal.IntList inlierIdMatchPos_ = emptyIntList();
        private Internal.IntList inlierLength_ = emptyIntList();
        private Internal.IntList outlierIds_ = emptyIntList();
        private Internal.IntList outlierIdMatchPos_ = emptyIntList();

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/TrackingProto$MotionBoxState$QuadOrBuilder.class */
        public interface QuadOrBuilder extends MessageLiteOrBuilder {
            List<Float> getVerticesList();

            int getVerticesCount();

            float getVertices(int index);
        }

        private MotionBoxState() {
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/TrackingProto$MotionBoxState$TrackStatus.class */
        public enum TrackStatus implements Internal.EnumLite {
            BOX_UNTRACKED(0),
            BOX_EMPTY(1),
            BOX_NO_FEATURES(2),
            BOX_TRACKED(3),
            BOX_DUPLICATED(4),
            BOX_TRACKED_OUT_OF_BOUND(5);
            
            public static final int BOX_UNTRACKED_VALUE = 0;
            public static final int BOX_EMPTY_VALUE = 1;
            public static final int BOX_NO_FEATURES_VALUE = 2;
            public static final int BOX_TRACKED_VALUE = 3;
            public static final int BOX_DUPLICATED_VALUE = 4;
            public static final int BOX_TRACKED_OUT_OF_BOUND_VALUE = 5;
            private static final Internal.EnumLiteMap<TrackStatus> internalValueMap = new Internal.EnumLiteMap<TrackStatus>() { // from class: com.google.mediapipe.tracking.TrackingProto.MotionBoxState.TrackStatus.1
                /* JADX DEBUG: Method merged with bridge method */
                /* JADX WARN: Can't rename method to resolve collision */
                @Override // com.google.protobuf.Internal.EnumLiteMap
                public TrackStatus findValueByNumber(int number) {
                    return TrackStatus.forNumber(number);
                }
            };
            private final int value;

            @Override // com.google.protobuf.Internal.EnumLite
            public final int getNumber() {
                return this.value;
            }

            @Deprecated
            public static TrackStatus valueOf(int value) {
                return forNumber(value);
            }

            public static TrackStatus forNumber(int value) {
                switch (value) {
                    case 0:
                        return BOX_UNTRACKED;
                    case 1:
                        return BOX_EMPTY;
                    case 2:
                        return BOX_NO_FEATURES;
                    case 3:
                        return BOX_TRACKED;
                    case 4:
                        return BOX_DUPLICATED;
                    case 5:
                        return BOX_TRACKED_OUT_OF_BOUND;
                    default:
                        return null;
                }
            }

            public static Internal.EnumLiteMap<TrackStatus> internalGetValueMap() {
                return internalValueMap;
            }

            public static Internal.EnumVerifier internalGetVerifier() {
                return TrackStatusVerifier.INSTANCE;
            }

            /* JADX INFO: Access modifiers changed from: private */
            /* JADX WARN: Classes with same name are omitted:
              classes2.dex
             */
            /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/TrackingProto$MotionBoxState$TrackStatus$TrackStatusVerifier.class */
            public static final class TrackStatusVerifier implements Internal.EnumVerifier {
                static final Internal.EnumVerifier INSTANCE = new TrackStatusVerifier();

                private TrackStatusVerifier() {
                }

                @Override // com.google.protobuf.Internal.EnumVerifier
                public boolean isInRange(int number) {
                    return TrackStatus.forNumber(number) != null;
                }
            }

            TrackStatus(int value) {
                this.value = value;
            }
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/TrackingProto$MotionBoxState$Quad.class */
        public static final class Quad extends GeneratedMessageLite<Quad, Builder> implements QuadOrBuilder {
            public static final int VERTICES_FIELD_NUMBER = 1;
            private Internal.FloatList vertices_ = emptyFloatList();
            private static final Quad DEFAULT_INSTANCE;
            private static volatile Parser<Quad> PARSER;

            private Quad() {
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxState.QuadOrBuilder
            public List<Float> getVerticesList() {
                return this.vertices_;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxState.QuadOrBuilder
            public int getVerticesCount() {
                return this.vertices_.size();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxState.QuadOrBuilder
            public float getVertices(int index) {
                return this.vertices_.getFloat(index);
            }

            private void ensureVerticesIsMutable() {
                if (!this.vertices_.isModifiable()) {
                    this.vertices_ = GeneratedMessageLite.mutableCopy(this.vertices_);
                }
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setVertices(int index, float value) {
                ensureVerticesIsMutable();
                this.vertices_.setFloat(index, value);
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void addVertices(float value) {
                ensureVerticesIsMutable();
                this.vertices_.addFloat(value);
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void addAllVertices(Iterable<? extends Float> values) {
                ensureVerticesIsMutable();
                AbstractMessageLite.addAll((Iterable) values, (List) this.vertices_);
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void clearVertices() {
                this.vertices_ = emptyFloatList();
            }

            public static Quad parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
                return (Quad) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
            }

            public static Quad parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
                return (Quad) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
            }

            public static Quad parseFrom(ByteString data) throws InvalidProtocolBufferException {
                return (Quad) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
            }

            public static Quad parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
                return (Quad) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
            }

            public static Quad parseFrom(byte[] data) throws InvalidProtocolBufferException {
                return (Quad) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
            }

            public static Quad parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
                return (Quad) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
            }

            public static Quad parseFrom(InputStream input) throws IOException {
                return (Quad) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
            }

            public static Quad parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
                return (Quad) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
            }

            public static Quad parseDelimitedFrom(InputStream input) throws IOException {
                return (Quad) parseDelimitedFrom(DEFAULT_INSTANCE, input);
            }

            public static Quad parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
                return (Quad) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
            }

            public static Quad parseFrom(CodedInputStream input) throws IOException {
                return (Quad) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
            }

            public static Quad parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
                return (Quad) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
            }

            public static Builder newBuilder() {
                return DEFAULT_INSTANCE.createBuilder();
            }

            public static Builder newBuilder(Quad prototype) {
                return DEFAULT_INSTANCE.createBuilder(prototype);
            }

            /* JADX WARN: Classes with same name are omitted:
              classes2.dex
             */
            /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/TrackingProto$MotionBoxState$Quad$Builder.class */
            public static final class Builder extends GeneratedMessageLite.Builder<Quad, Builder> implements QuadOrBuilder {
                private Builder() {
                    super(Quad.DEFAULT_INSTANCE);
                }

                @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxState.QuadOrBuilder
                public List<Float> getVerticesList() {
                    return Collections.unmodifiableList(((Quad) this.instance).getVerticesList());
                }

                @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxState.QuadOrBuilder
                public int getVerticesCount() {
                    return ((Quad) this.instance).getVerticesCount();
                }

                @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxState.QuadOrBuilder
                public float getVertices(int index) {
                    return ((Quad) this.instance).getVertices(index);
                }

                public Builder setVertices(int index, float value) {
                    copyOnWrite();
                    ((Quad) this.instance).setVertices(index, value);
                    return this;
                }

                public Builder addVertices(float value) {
                    copyOnWrite();
                    ((Quad) this.instance).addVertices(value);
                    return this;
                }

                public Builder addAllVertices(Iterable<? extends Float> values) {
                    copyOnWrite();
                    ((Quad) this.instance).addAllVertices(values);
                    return this;
                }

                public Builder clearVertices() {
                    copyOnWrite();
                    ((Quad) this.instance).clearVertices();
                    return this;
                }
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.google.protobuf.GeneratedMessageLite
            public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
                switch (method) {
                    case NEW_MUTABLE_INSTANCE:
                        return new Quad();
                    case NEW_BUILDER:
                        return new Builder();
                    case BUILD_MESSAGE_INFO:
                        Object[] objects = {"vertices_"};
                        return newMessageInfo(DEFAULT_INSTANCE, "\u0001\u0001����\u0001\u0001\u0001��\u0001��\u0001\u0013", objects);
                    case GET_DEFAULT_INSTANCE:
                        return DEFAULT_INSTANCE;
                    case GET_PARSER:
                        Parser<Quad> parser = PARSER;
                        if (parser == null) {
                            synchronized (Quad.class) {
                                parser = PARSER;
                                if (parser == null) {
                                    parser = new GeneratedMessageLite.DefaultInstanceBasedParser<>(DEFAULT_INSTANCE);
                                    PARSER = parser;
                                }
                            }
                        }
                        return parser;
                    case GET_MEMOIZED_IS_INITIALIZED:
                        return (byte) 1;
                    case SET_MEMOIZED_IS_INITIALIZED:
                        return null;
                    default:
                        throw new UnsupportedOperationException();
                }
            }

            static {
                Quad defaultInstance = new Quad();
                DEFAULT_INSTANCE = defaultInstance;
                GeneratedMessageLite.registerDefaultInstance(Quad.class, defaultInstance);
            }

            public static Quad getDefaultInstance() {
                return DEFAULT_INSTANCE;
            }

            public static Parser<Quad> parser() {
                return DEFAULT_INSTANCE.getParserForType();
            }
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public boolean hasPosX() {
            return (this.bitField0_ & 1) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public float getPosX() {
            return this.posX_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setPosX(float value) {
            this.bitField0_ |= 1;
            this.posX_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearPosX() {
            this.bitField0_ &= -2;
            this.posX_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public boolean hasPosY() {
            return (this.bitField0_ & 2) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public float getPosY() {
            return this.posY_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setPosY(float value) {
            this.bitField0_ |= 2;
            this.posY_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearPosY() {
            this.bitField0_ &= -3;
            this.posY_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public boolean hasWidth() {
            return (this.bitField0_ & 4) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public float getWidth() {
            return this.width_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setWidth(float value) {
            this.bitField0_ |= 4;
            this.width_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearWidth() {
            this.bitField0_ &= -5;
            this.width_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public boolean hasHeight() {
            return (this.bitField0_ & 8) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public float getHeight() {
            return this.height_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setHeight(float value) {
            this.bitField0_ |= 8;
            this.height_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearHeight() {
            this.bitField0_ &= -9;
            this.height_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public boolean hasScale() {
            return (this.bitField0_ & 16) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public float getScale() {
            return this.scale_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setScale(float value) {
            this.bitField0_ |= 16;
            this.scale_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearScale() {
            this.bitField0_ &= -17;
            this.scale_ = 1.0f;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public boolean hasRotation() {
            return (this.bitField0_ & 32) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public float getRotation() {
            return this.rotation_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setRotation(float value) {
            this.bitField0_ |= 32;
            this.rotation_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearRotation() {
            this.bitField0_ &= -33;
            this.rotation_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public boolean hasQuad() {
            return (this.bitField0_ & 64) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public Quad getQuad() {
            return this.quad_ == null ? Quad.getDefaultInstance() : this.quad_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setQuad(Quad value) {
            value.getClass();
            this.quad_ = value;
            this.bitField0_ |= 64;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void mergeQuad(Quad value) {
            value.getClass();
            if (this.quad_ != null && this.quad_ != Quad.getDefaultInstance()) {
                this.quad_ = Quad.newBuilder(this.quad_).mergeFrom((Quad.Builder) value).buildPartial();
            } else {
                this.quad_ = value;
            }
            this.bitField0_ |= 64;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearQuad() {
            this.quad_ = null;
            this.bitField0_ &= -65;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public boolean hasAspectRatio() {
            return (this.bitField0_ & 128) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public float getAspectRatio() {
            return this.aspectRatio_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setAspectRatio(float value) {
            this.bitField0_ |= 128;
            this.aspectRatio_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearAspectRatio() {
            this.bitField0_ &= -129;
            this.aspectRatio_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public boolean hasRequestGrouping() {
            return (this.bitField0_ & 256) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public boolean getRequestGrouping() {
            return this.requestGrouping_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setRequestGrouping(boolean value) {
            this.bitField0_ |= 256;
            this.requestGrouping_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearRequestGrouping() {
            this.bitField0_ &= -257;
            this.requestGrouping_ = false;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public boolean hasPnpHomography() {
            return (this.bitField0_ & 512) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public MotionModelsProto.Homography getPnpHomography() {
            return this.pnpHomography_ == null ? MotionModelsProto.Homography.getDefaultInstance() : this.pnpHomography_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setPnpHomography(MotionModelsProto.Homography value) {
            value.getClass();
            this.pnpHomography_ = value;
            this.bitField0_ |= 512;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void mergePnpHomography(MotionModelsProto.Homography value) {
            value.getClass();
            if (this.pnpHomography_ != null && this.pnpHomography_ != MotionModelsProto.Homography.getDefaultInstance()) {
                this.pnpHomography_ = MotionModelsProto.Homography.newBuilder(this.pnpHomography_).mergeFrom((MotionModelsProto.Homography.Builder) value).buildPartial();
            } else {
                this.pnpHomography_ = value;
            }
            this.bitField0_ |= 512;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearPnpHomography() {
            this.pnpHomography_ = null;
            this.bitField0_ &= -513;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public boolean hasDx() {
            return (this.bitField0_ & 1024) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public float getDx() {
            return this.dx_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setDx(float value) {
            this.bitField0_ |= 1024;
            this.dx_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearDx() {
            this.bitField0_ &= -1025;
            this.dx_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public boolean hasDy() {
            return (this.bitField0_ & 2048) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public float getDy() {
            return this.dy_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setDy(float value) {
            this.bitField0_ |= 2048;
            this.dy_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearDy() {
            this.bitField0_ &= -2049;
            this.dy_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public boolean hasKineticEnergy() {
            return (this.bitField0_ & 4096) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public float getKineticEnergy() {
            return this.kineticEnergy_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setKineticEnergy(float value) {
            this.bitField0_ |= 4096;
            this.kineticEnergy_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearKineticEnergy() {
            this.bitField0_ &= -4097;
            this.kineticEnergy_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public boolean hasPriorWeight() {
            return (this.bitField0_ & 8192) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public float getPriorWeight() {
            return this.priorWeight_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setPriorWeight(float value) {
            this.bitField0_ |= 8192;
            this.priorWeight_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearPriorWeight() {
            this.bitField0_ &= -8193;
            this.priorWeight_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public boolean hasTrackStatus() {
            return (this.bitField0_ & Calib3d.CALIB_RATIONAL_MODEL) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public TrackStatus getTrackStatus() {
            TrackStatus result = TrackStatus.forNumber(this.trackStatus_);
            return result == null ? TrackStatus.BOX_UNTRACKED : result;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setTrackStatus(TrackStatus value) {
            this.trackStatus_ = value.getNumber();
            this.bitField0_ |= Calib3d.CALIB_RATIONAL_MODEL;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearTrackStatus() {
            this.bitField0_ &= -16385;
            this.trackStatus_ = 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public boolean hasSpatialPriorGridSize() {
            return (this.bitField0_ & Calib3d.CALIB_THIN_PRISM_MODEL) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public int getSpatialPriorGridSize() {
            return this.spatialPriorGridSize_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setSpatialPriorGridSize(int value) {
            this.bitField0_ |= Calib3d.CALIB_THIN_PRISM_MODEL;
            this.spatialPriorGridSize_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearSpatialPriorGridSize() {
            this.bitField0_ &= -32769;
            this.spatialPriorGridSize_ = 10;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public List<Float> getSpatialPriorList() {
            return this.spatialPrior_;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public int getSpatialPriorCount() {
            return this.spatialPrior_.size();
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public float getSpatialPrior(int index) {
            return this.spatialPrior_.getFloat(index);
        }

        private void ensureSpatialPriorIsMutable() {
            if (!this.spatialPrior_.isModifiable()) {
                this.spatialPrior_ = GeneratedMessageLite.mutableCopy(this.spatialPrior_);
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setSpatialPrior(int index, float value) {
            ensureSpatialPriorIsMutable();
            this.spatialPrior_.setFloat(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addSpatialPrior(float value) {
            ensureSpatialPriorIsMutable();
            this.spatialPrior_.addFloat(value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addAllSpatialPrior(Iterable<? extends Float> values) {
            ensureSpatialPriorIsMutable();
            AbstractMessageLite.addAll((Iterable) values, (List) this.spatialPrior_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearSpatialPrior() {
            this.spatialPrior_ = emptyFloatList();
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public List<Float> getSpatialConfidenceList() {
            return this.spatialConfidence_;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public int getSpatialConfidenceCount() {
            return this.spatialConfidence_.size();
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public float getSpatialConfidence(int index) {
            return this.spatialConfidence_.getFloat(index);
        }

        private void ensureSpatialConfidenceIsMutable() {
            if (!this.spatialConfidence_.isModifiable()) {
                this.spatialConfidence_ = GeneratedMessageLite.mutableCopy(this.spatialConfidence_);
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setSpatialConfidence(int index, float value) {
            ensureSpatialConfidenceIsMutable();
            this.spatialConfidence_.setFloat(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addSpatialConfidence(float value) {
            ensureSpatialConfidenceIsMutable();
            this.spatialConfidence_.addFloat(value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addAllSpatialConfidence(Iterable<? extends Float> values) {
            ensureSpatialConfidenceIsMutable();
            AbstractMessageLite.addAll((Iterable) values, (List) this.spatialConfidence_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearSpatialConfidence() {
            this.spatialConfidence_ = emptyFloatList();
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public boolean hasPriorDiff() {
            return (this.bitField0_ & 65536) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public float getPriorDiff() {
            return this.priorDiff_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setPriorDiff(float value) {
            this.bitField0_ |= 65536;
            this.priorDiff_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearPriorDiff() {
            this.bitField0_ &= -65537;
            this.priorDiff_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public boolean hasMotionDisparity() {
            return (this.bitField0_ & 131072) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public float getMotionDisparity() {
            return this.motionDisparity_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setMotionDisparity(float value) {
            this.bitField0_ |= 131072;
            this.motionDisparity_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearMotionDisparity() {
            this.bitField0_ &= -131073;
            this.motionDisparity_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public boolean hasBackgroundDiscrimination() {
            return (this.bitField0_ & Calib3d.CALIB_TILTED_MODEL) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public float getBackgroundDiscrimination() {
            return this.backgroundDiscrimination_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setBackgroundDiscrimination(float value) {
            this.bitField0_ |= Calib3d.CALIB_TILTED_MODEL;
            this.backgroundDiscrimination_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearBackgroundDiscrimination() {
            this.bitField0_ &= -262145;
            this.backgroundDiscrimination_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public boolean hasInlierCenterX() {
            return (this.bitField0_ & 524288) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public float getInlierCenterX() {
            return this.inlierCenterX_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setInlierCenterX(float value) {
            this.bitField0_ |= 524288;
            this.inlierCenterX_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearInlierCenterX() {
            this.bitField0_ &= -524289;
            this.inlierCenterX_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public boolean hasInlierCenterY() {
            return (this.bitField0_ & 1048576) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public float getInlierCenterY() {
            return this.inlierCenterY_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setInlierCenterY(float value) {
            this.bitField0_ |= 1048576;
            this.inlierCenterY_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearInlierCenterY() {
            this.bitField0_ &= -1048577;
            this.inlierCenterY_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public boolean hasInlierSum() {
            return (this.bitField0_ & Calib3d.CALIB_FIX_TANGENT_DIST) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public float getInlierSum() {
            return this.inlierSum_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setInlierSum(float value) {
            this.bitField0_ |= Calib3d.CALIB_FIX_TANGENT_DIST;
            this.inlierSum_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearInlierSum() {
            this.bitField0_ &= -2097153;
            this.inlierSum_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public boolean hasInlierRatio() {
            return (this.bitField0_ & Calib3d.CALIB_USE_EXTRINSIC_GUESS) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public float getInlierRatio() {
            return this.inlierRatio_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setInlierRatio(float value) {
            this.bitField0_ |= Calib3d.CALIB_USE_EXTRINSIC_GUESS;
            this.inlierRatio_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearInlierRatio() {
            this.bitField0_ &= -4194305;
            this.inlierRatio_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public boolean hasInlierWidth() {
            return (this.bitField0_ & 8388608) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public float getInlierWidth() {
            return this.inlierWidth_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setInlierWidth(float value) {
            this.bitField0_ |= 8388608;
            this.inlierWidth_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearInlierWidth() {
            this.bitField0_ &= -8388609;
            this.inlierWidth_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public boolean hasInlierHeight() {
            return (this.bitField0_ & 16777216) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public float getInlierHeight() {
            return this.inlierHeight_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setInlierHeight(float value) {
            this.bitField0_ |= 16777216;
            this.inlierHeight_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearInlierHeight() {
            this.bitField0_ &= -16777217;
            this.inlierHeight_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public List<Integer> getInlierIdsList() {
            return this.inlierIds_;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public int getInlierIdsCount() {
            return this.inlierIds_.size();
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public int getInlierIds(int index) {
            return this.inlierIds_.getInt(index);
        }

        private void ensureInlierIdsIsMutable() {
            if (!this.inlierIds_.isModifiable()) {
                this.inlierIds_ = GeneratedMessageLite.mutableCopy(this.inlierIds_);
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setInlierIds(int index, int value) {
            ensureInlierIdsIsMutable();
            this.inlierIds_.setInt(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addInlierIds(int value) {
            ensureInlierIdsIsMutable();
            this.inlierIds_.addInt(value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addAllInlierIds(Iterable<? extends Integer> values) {
            ensureInlierIdsIsMutable();
            AbstractMessageLite.addAll((Iterable) values, (List) this.inlierIds_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearInlierIds() {
            this.inlierIds_ = emptyIntList();
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public List<Integer> getInlierIdMatchPosList() {
            return this.inlierIdMatchPos_;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public int getInlierIdMatchPosCount() {
            return this.inlierIdMatchPos_.size();
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public int getInlierIdMatchPos(int index) {
            return this.inlierIdMatchPos_.getInt(index);
        }

        private void ensureInlierIdMatchPosIsMutable() {
            if (!this.inlierIdMatchPos_.isModifiable()) {
                this.inlierIdMatchPos_ = GeneratedMessageLite.mutableCopy(this.inlierIdMatchPos_);
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setInlierIdMatchPos(int index, int value) {
            ensureInlierIdMatchPosIsMutable();
            this.inlierIdMatchPos_.setInt(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addInlierIdMatchPos(int value) {
            ensureInlierIdMatchPosIsMutable();
            this.inlierIdMatchPos_.addInt(value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addAllInlierIdMatchPos(Iterable<? extends Integer> values) {
            ensureInlierIdMatchPosIsMutable();
            AbstractMessageLite.addAll((Iterable) values, (List) this.inlierIdMatchPos_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearInlierIdMatchPos() {
            this.inlierIdMatchPos_ = emptyIntList();
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public List<Integer> getInlierLengthList() {
            return this.inlierLength_;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public int getInlierLengthCount() {
            return this.inlierLength_.size();
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public int getInlierLength(int index) {
            return this.inlierLength_.getInt(index);
        }

        private void ensureInlierLengthIsMutable() {
            if (!this.inlierLength_.isModifiable()) {
                this.inlierLength_ = GeneratedMessageLite.mutableCopy(this.inlierLength_);
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setInlierLength(int index, int value) {
            ensureInlierLengthIsMutable();
            this.inlierLength_.setInt(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addInlierLength(int value) {
            ensureInlierLengthIsMutable();
            this.inlierLength_.addInt(value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addAllInlierLength(Iterable<? extends Integer> values) {
            ensureInlierLengthIsMutable();
            AbstractMessageLite.addAll((Iterable) values, (List) this.inlierLength_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearInlierLength() {
            this.inlierLength_ = emptyIntList();
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public List<Integer> getOutlierIdsList() {
            return this.outlierIds_;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public int getOutlierIdsCount() {
            return this.outlierIds_.size();
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public int getOutlierIds(int index) {
            return this.outlierIds_.getInt(index);
        }

        private void ensureOutlierIdsIsMutable() {
            if (!this.outlierIds_.isModifiable()) {
                this.outlierIds_ = GeneratedMessageLite.mutableCopy(this.outlierIds_);
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setOutlierIds(int index, int value) {
            ensureOutlierIdsIsMutable();
            this.outlierIds_.setInt(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addOutlierIds(int value) {
            ensureOutlierIdsIsMutable();
            this.outlierIds_.addInt(value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addAllOutlierIds(Iterable<? extends Integer> values) {
            ensureOutlierIdsIsMutable();
            AbstractMessageLite.addAll((Iterable) values, (List) this.outlierIds_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearOutlierIds() {
            this.outlierIds_ = emptyIntList();
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public List<Integer> getOutlierIdMatchPosList() {
            return this.outlierIdMatchPos_;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public int getOutlierIdMatchPosCount() {
            return this.outlierIdMatchPos_.size();
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public int getOutlierIdMatchPos(int index) {
            return this.outlierIdMatchPos_.getInt(index);
        }

        private void ensureOutlierIdMatchPosIsMutable() {
            if (!this.outlierIdMatchPos_.isModifiable()) {
                this.outlierIdMatchPos_ = GeneratedMessageLite.mutableCopy(this.outlierIdMatchPos_);
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setOutlierIdMatchPos(int index, int value) {
            ensureOutlierIdMatchPosIsMutable();
            this.outlierIdMatchPos_.setInt(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addOutlierIdMatchPos(int value) {
            ensureOutlierIdMatchPosIsMutable();
            this.outlierIdMatchPos_.addInt(value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addAllOutlierIdMatchPos(Iterable<? extends Integer> values) {
            ensureOutlierIdMatchPosIsMutable();
            AbstractMessageLite.addAll((Iterable) values, (List) this.outlierIdMatchPos_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearOutlierIdMatchPos() {
            this.outlierIdMatchPos_ = emptyIntList();
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public boolean hasTrackingConfidence() {
            return (this.bitField0_ & 33554432) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public float getTrackingConfidence() {
            return this.trackingConfidence_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setTrackingConfidence(float value) {
            this.bitField0_ |= 33554432;
            this.trackingConfidence_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearTrackingConfidence() {
            this.bitField0_ &= -33554433;
            this.trackingConfidence_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public boolean hasInternal() {
            return (this.bitField0_ & 67108864) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
        public MotionBoxInternalState getInternal() {
            return this.internal_ == null ? MotionBoxInternalState.getDefaultInstance() : this.internal_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setInternal(MotionBoxInternalState value) {
            value.getClass();
            this.internal_ = value;
            this.bitField0_ |= 67108864;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void mergeInternal(MotionBoxInternalState value) {
            value.getClass();
            if (this.internal_ != null && this.internal_ != MotionBoxInternalState.getDefaultInstance()) {
                this.internal_ = MotionBoxInternalState.newBuilder(this.internal_).mergeFrom((MotionBoxInternalState.Builder) value).buildPartial();
            } else {
                this.internal_ = value;
            }
            this.bitField0_ |= 67108864;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearInternal() {
            this.internal_ = null;
            this.bitField0_ &= -67108865;
        }

        public static MotionBoxState parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
            return (MotionBoxState) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static MotionBoxState parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (MotionBoxState) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static MotionBoxState parseFrom(ByteString data) throws InvalidProtocolBufferException {
            return (MotionBoxState) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static MotionBoxState parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (MotionBoxState) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static MotionBoxState parseFrom(byte[] data) throws InvalidProtocolBufferException {
            return (MotionBoxState) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static MotionBoxState parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (MotionBoxState) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static MotionBoxState parseFrom(InputStream input) throws IOException {
            return (MotionBoxState) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static MotionBoxState parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (MotionBoxState) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static MotionBoxState parseDelimitedFrom(InputStream input) throws IOException {
            return (MotionBoxState) parseDelimitedFrom(DEFAULT_INSTANCE, input);
        }

        public static MotionBoxState parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (MotionBoxState) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static MotionBoxState parseFrom(CodedInputStream input) throws IOException {
            return (MotionBoxState) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static MotionBoxState parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (MotionBoxState) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Builder newBuilder() {
            return DEFAULT_INSTANCE.createBuilder();
        }

        public static Builder newBuilder(MotionBoxState prototype) {
            return DEFAULT_INSTANCE.createBuilder(prototype);
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/TrackingProto$MotionBoxState$Builder.class */
        public static final class Builder extends GeneratedMessageLite.Builder<MotionBoxState, Builder> implements MotionBoxStateOrBuilder {
            private Builder() {
                super(MotionBoxState.DEFAULT_INSTANCE);
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public boolean hasPosX() {
                return ((MotionBoxState) this.instance).hasPosX();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public float getPosX() {
                return ((MotionBoxState) this.instance).getPosX();
            }

            public Builder setPosX(float value) {
                copyOnWrite();
                ((MotionBoxState) this.instance).setPosX(value);
                return this;
            }

            public Builder clearPosX() {
                copyOnWrite();
                ((MotionBoxState) this.instance).clearPosX();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public boolean hasPosY() {
                return ((MotionBoxState) this.instance).hasPosY();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public float getPosY() {
                return ((MotionBoxState) this.instance).getPosY();
            }

            public Builder setPosY(float value) {
                copyOnWrite();
                ((MotionBoxState) this.instance).setPosY(value);
                return this;
            }

            public Builder clearPosY() {
                copyOnWrite();
                ((MotionBoxState) this.instance).clearPosY();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public boolean hasWidth() {
                return ((MotionBoxState) this.instance).hasWidth();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public float getWidth() {
                return ((MotionBoxState) this.instance).getWidth();
            }

            public Builder setWidth(float value) {
                copyOnWrite();
                ((MotionBoxState) this.instance).setWidth(value);
                return this;
            }

            public Builder clearWidth() {
                copyOnWrite();
                ((MotionBoxState) this.instance).clearWidth();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public boolean hasHeight() {
                return ((MotionBoxState) this.instance).hasHeight();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public float getHeight() {
                return ((MotionBoxState) this.instance).getHeight();
            }

            public Builder setHeight(float value) {
                copyOnWrite();
                ((MotionBoxState) this.instance).setHeight(value);
                return this;
            }

            public Builder clearHeight() {
                copyOnWrite();
                ((MotionBoxState) this.instance).clearHeight();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public boolean hasScale() {
                return ((MotionBoxState) this.instance).hasScale();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public float getScale() {
                return ((MotionBoxState) this.instance).getScale();
            }

            public Builder setScale(float value) {
                copyOnWrite();
                ((MotionBoxState) this.instance).setScale(value);
                return this;
            }

            public Builder clearScale() {
                copyOnWrite();
                ((MotionBoxState) this.instance).clearScale();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public boolean hasRotation() {
                return ((MotionBoxState) this.instance).hasRotation();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public float getRotation() {
                return ((MotionBoxState) this.instance).getRotation();
            }

            public Builder setRotation(float value) {
                copyOnWrite();
                ((MotionBoxState) this.instance).setRotation(value);
                return this;
            }

            public Builder clearRotation() {
                copyOnWrite();
                ((MotionBoxState) this.instance).clearRotation();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public boolean hasQuad() {
                return ((MotionBoxState) this.instance).hasQuad();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public Quad getQuad() {
                return ((MotionBoxState) this.instance).getQuad();
            }

            public Builder setQuad(Quad value) {
                copyOnWrite();
                ((MotionBoxState) this.instance).setQuad(value);
                return this;
            }

            public Builder setQuad(Quad.Builder builderForValue) {
                copyOnWrite();
                ((MotionBoxState) this.instance).setQuad(builderForValue.build());
                return this;
            }

            public Builder mergeQuad(Quad value) {
                copyOnWrite();
                ((MotionBoxState) this.instance).mergeQuad(value);
                return this;
            }

            public Builder clearQuad() {
                copyOnWrite();
                ((MotionBoxState) this.instance).clearQuad();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public boolean hasAspectRatio() {
                return ((MotionBoxState) this.instance).hasAspectRatio();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public float getAspectRatio() {
                return ((MotionBoxState) this.instance).getAspectRatio();
            }

            public Builder setAspectRatio(float value) {
                copyOnWrite();
                ((MotionBoxState) this.instance).setAspectRatio(value);
                return this;
            }

            public Builder clearAspectRatio() {
                copyOnWrite();
                ((MotionBoxState) this.instance).clearAspectRatio();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public boolean hasRequestGrouping() {
                return ((MotionBoxState) this.instance).hasRequestGrouping();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public boolean getRequestGrouping() {
                return ((MotionBoxState) this.instance).getRequestGrouping();
            }

            public Builder setRequestGrouping(boolean value) {
                copyOnWrite();
                ((MotionBoxState) this.instance).setRequestGrouping(value);
                return this;
            }

            public Builder clearRequestGrouping() {
                copyOnWrite();
                ((MotionBoxState) this.instance).clearRequestGrouping();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public boolean hasPnpHomography() {
                return ((MotionBoxState) this.instance).hasPnpHomography();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public MotionModelsProto.Homography getPnpHomography() {
                return ((MotionBoxState) this.instance).getPnpHomography();
            }

            public Builder setPnpHomography(MotionModelsProto.Homography value) {
                copyOnWrite();
                ((MotionBoxState) this.instance).setPnpHomography(value);
                return this;
            }

            public Builder setPnpHomography(MotionModelsProto.Homography.Builder builderForValue) {
                copyOnWrite();
                ((MotionBoxState) this.instance).setPnpHomography(builderForValue.build());
                return this;
            }

            public Builder mergePnpHomography(MotionModelsProto.Homography value) {
                copyOnWrite();
                ((MotionBoxState) this.instance).mergePnpHomography(value);
                return this;
            }

            public Builder clearPnpHomography() {
                copyOnWrite();
                ((MotionBoxState) this.instance).clearPnpHomography();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public boolean hasDx() {
                return ((MotionBoxState) this.instance).hasDx();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public float getDx() {
                return ((MotionBoxState) this.instance).getDx();
            }

            public Builder setDx(float value) {
                copyOnWrite();
                ((MotionBoxState) this.instance).setDx(value);
                return this;
            }

            public Builder clearDx() {
                copyOnWrite();
                ((MotionBoxState) this.instance).clearDx();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public boolean hasDy() {
                return ((MotionBoxState) this.instance).hasDy();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public float getDy() {
                return ((MotionBoxState) this.instance).getDy();
            }

            public Builder setDy(float value) {
                copyOnWrite();
                ((MotionBoxState) this.instance).setDy(value);
                return this;
            }

            public Builder clearDy() {
                copyOnWrite();
                ((MotionBoxState) this.instance).clearDy();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public boolean hasKineticEnergy() {
                return ((MotionBoxState) this.instance).hasKineticEnergy();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public float getKineticEnergy() {
                return ((MotionBoxState) this.instance).getKineticEnergy();
            }

            public Builder setKineticEnergy(float value) {
                copyOnWrite();
                ((MotionBoxState) this.instance).setKineticEnergy(value);
                return this;
            }

            public Builder clearKineticEnergy() {
                copyOnWrite();
                ((MotionBoxState) this.instance).clearKineticEnergy();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public boolean hasPriorWeight() {
                return ((MotionBoxState) this.instance).hasPriorWeight();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public float getPriorWeight() {
                return ((MotionBoxState) this.instance).getPriorWeight();
            }

            public Builder setPriorWeight(float value) {
                copyOnWrite();
                ((MotionBoxState) this.instance).setPriorWeight(value);
                return this;
            }

            public Builder clearPriorWeight() {
                copyOnWrite();
                ((MotionBoxState) this.instance).clearPriorWeight();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public boolean hasTrackStatus() {
                return ((MotionBoxState) this.instance).hasTrackStatus();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public TrackStatus getTrackStatus() {
                return ((MotionBoxState) this.instance).getTrackStatus();
            }

            public Builder setTrackStatus(TrackStatus value) {
                copyOnWrite();
                ((MotionBoxState) this.instance).setTrackStatus(value);
                return this;
            }

            public Builder clearTrackStatus() {
                copyOnWrite();
                ((MotionBoxState) this.instance).clearTrackStatus();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public boolean hasSpatialPriorGridSize() {
                return ((MotionBoxState) this.instance).hasSpatialPriorGridSize();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public int getSpatialPriorGridSize() {
                return ((MotionBoxState) this.instance).getSpatialPriorGridSize();
            }

            public Builder setSpatialPriorGridSize(int value) {
                copyOnWrite();
                ((MotionBoxState) this.instance).setSpatialPriorGridSize(value);
                return this;
            }

            public Builder clearSpatialPriorGridSize() {
                copyOnWrite();
                ((MotionBoxState) this.instance).clearSpatialPriorGridSize();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public List<Float> getSpatialPriorList() {
                return Collections.unmodifiableList(((MotionBoxState) this.instance).getSpatialPriorList());
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public int getSpatialPriorCount() {
                return ((MotionBoxState) this.instance).getSpatialPriorCount();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public float getSpatialPrior(int index) {
                return ((MotionBoxState) this.instance).getSpatialPrior(index);
            }

            public Builder setSpatialPrior(int index, float value) {
                copyOnWrite();
                ((MotionBoxState) this.instance).setSpatialPrior(index, value);
                return this;
            }

            public Builder addSpatialPrior(float value) {
                copyOnWrite();
                ((MotionBoxState) this.instance).addSpatialPrior(value);
                return this;
            }

            public Builder addAllSpatialPrior(Iterable<? extends Float> values) {
                copyOnWrite();
                ((MotionBoxState) this.instance).addAllSpatialPrior(values);
                return this;
            }

            public Builder clearSpatialPrior() {
                copyOnWrite();
                ((MotionBoxState) this.instance).clearSpatialPrior();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public List<Float> getSpatialConfidenceList() {
                return Collections.unmodifiableList(((MotionBoxState) this.instance).getSpatialConfidenceList());
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public int getSpatialConfidenceCount() {
                return ((MotionBoxState) this.instance).getSpatialConfidenceCount();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public float getSpatialConfidence(int index) {
                return ((MotionBoxState) this.instance).getSpatialConfidence(index);
            }

            public Builder setSpatialConfidence(int index, float value) {
                copyOnWrite();
                ((MotionBoxState) this.instance).setSpatialConfidence(index, value);
                return this;
            }

            public Builder addSpatialConfidence(float value) {
                copyOnWrite();
                ((MotionBoxState) this.instance).addSpatialConfidence(value);
                return this;
            }

            public Builder addAllSpatialConfidence(Iterable<? extends Float> values) {
                copyOnWrite();
                ((MotionBoxState) this.instance).addAllSpatialConfidence(values);
                return this;
            }

            public Builder clearSpatialConfidence() {
                copyOnWrite();
                ((MotionBoxState) this.instance).clearSpatialConfidence();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public boolean hasPriorDiff() {
                return ((MotionBoxState) this.instance).hasPriorDiff();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public float getPriorDiff() {
                return ((MotionBoxState) this.instance).getPriorDiff();
            }

            public Builder setPriorDiff(float value) {
                copyOnWrite();
                ((MotionBoxState) this.instance).setPriorDiff(value);
                return this;
            }

            public Builder clearPriorDiff() {
                copyOnWrite();
                ((MotionBoxState) this.instance).clearPriorDiff();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public boolean hasMotionDisparity() {
                return ((MotionBoxState) this.instance).hasMotionDisparity();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public float getMotionDisparity() {
                return ((MotionBoxState) this.instance).getMotionDisparity();
            }

            public Builder setMotionDisparity(float value) {
                copyOnWrite();
                ((MotionBoxState) this.instance).setMotionDisparity(value);
                return this;
            }

            public Builder clearMotionDisparity() {
                copyOnWrite();
                ((MotionBoxState) this.instance).clearMotionDisparity();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public boolean hasBackgroundDiscrimination() {
                return ((MotionBoxState) this.instance).hasBackgroundDiscrimination();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public float getBackgroundDiscrimination() {
                return ((MotionBoxState) this.instance).getBackgroundDiscrimination();
            }

            public Builder setBackgroundDiscrimination(float value) {
                copyOnWrite();
                ((MotionBoxState) this.instance).setBackgroundDiscrimination(value);
                return this;
            }

            public Builder clearBackgroundDiscrimination() {
                copyOnWrite();
                ((MotionBoxState) this.instance).clearBackgroundDiscrimination();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public boolean hasInlierCenterX() {
                return ((MotionBoxState) this.instance).hasInlierCenterX();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public float getInlierCenterX() {
                return ((MotionBoxState) this.instance).getInlierCenterX();
            }

            public Builder setInlierCenterX(float value) {
                copyOnWrite();
                ((MotionBoxState) this.instance).setInlierCenterX(value);
                return this;
            }

            public Builder clearInlierCenterX() {
                copyOnWrite();
                ((MotionBoxState) this.instance).clearInlierCenterX();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public boolean hasInlierCenterY() {
                return ((MotionBoxState) this.instance).hasInlierCenterY();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public float getInlierCenterY() {
                return ((MotionBoxState) this.instance).getInlierCenterY();
            }

            public Builder setInlierCenterY(float value) {
                copyOnWrite();
                ((MotionBoxState) this.instance).setInlierCenterY(value);
                return this;
            }

            public Builder clearInlierCenterY() {
                copyOnWrite();
                ((MotionBoxState) this.instance).clearInlierCenterY();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public boolean hasInlierSum() {
                return ((MotionBoxState) this.instance).hasInlierSum();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public float getInlierSum() {
                return ((MotionBoxState) this.instance).getInlierSum();
            }

            public Builder setInlierSum(float value) {
                copyOnWrite();
                ((MotionBoxState) this.instance).setInlierSum(value);
                return this;
            }

            public Builder clearInlierSum() {
                copyOnWrite();
                ((MotionBoxState) this.instance).clearInlierSum();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public boolean hasInlierRatio() {
                return ((MotionBoxState) this.instance).hasInlierRatio();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public float getInlierRatio() {
                return ((MotionBoxState) this.instance).getInlierRatio();
            }

            public Builder setInlierRatio(float value) {
                copyOnWrite();
                ((MotionBoxState) this.instance).setInlierRatio(value);
                return this;
            }

            public Builder clearInlierRatio() {
                copyOnWrite();
                ((MotionBoxState) this.instance).clearInlierRatio();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public boolean hasInlierWidth() {
                return ((MotionBoxState) this.instance).hasInlierWidth();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public float getInlierWidth() {
                return ((MotionBoxState) this.instance).getInlierWidth();
            }

            public Builder setInlierWidth(float value) {
                copyOnWrite();
                ((MotionBoxState) this.instance).setInlierWidth(value);
                return this;
            }

            public Builder clearInlierWidth() {
                copyOnWrite();
                ((MotionBoxState) this.instance).clearInlierWidth();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public boolean hasInlierHeight() {
                return ((MotionBoxState) this.instance).hasInlierHeight();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public float getInlierHeight() {
                return ((MotionBoxState) this.instance).getInlierHeight();
            }

            public Builder setInlierHeight(float value) {
                copyOnWrite();
                ((MotionBoxState) this.instance).setInlierHeight(value);
                return this;
            }

            public Builder clearInlierHeight() {
                copyOnWrite();
                ((MotionBoxState) this.instance).clearInlierHeight();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public List<Integer> getInlierIdsList() {
                return Collections.unmodifiableList(((MotionBoxState) this.instance).getInlierIdsList());
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public int getInlierIdsCount() {
                return ((MotionBoxState) this.instance).getInlierIdsCount();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public int getInlierIds(int index) {
                return ((MotionBoxState) this.instance).getInlierIds(index);
            }

            public Builder setInlierIds(int index, int value) {
                copyOnWrite();
                ((MotionBoxState) this.instance).setInlierIds(index, value);
                return this;
            }

            public Builder addInlierIds(int value) {
                copyOnWrite();
                ((MotionBoxState) this.instance).addInlierIds(value);
                return this;
            }

            public Builder addAllInlierIds(Iterable<? extends Integer> values) {
                copyOnWrite();
                ((MotionBoxState) this.instance).addAllInlierIds(values);
                return this;
            }

            public Builder clearInlierIds() {
                copyOnWrite();
                ((MotionBoxState) this.instance).clearInlierIds();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public List<Integer> getInlierIdMatchPosList() {
                return Collections.unmodifiableList(((MotionBoxState) this.instance).getInlierIdMatchPosList());
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public int getInlierIdMatchPosCount() {
                return ((MotionBoxState) this.instance).getInlierIdMatchPosCount();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public int getInlierIdMatchPos(int index) {
                return ((MotionBoxState) this.instance).getInlierIdMatchPos(index);
            }

            public Builder setInlierIdMatchPos(int index, int value) {
                copyOnWrite();
                ((MotionBoxState) this.instance).setInlierIdMatchPos(index, value);
                return this;
            }

            public Builder addInlierIdMatchPos(int value) {
                copyOnWrite();
                ((MotionBoxState) this.instance).addInlierIdMatchPos(value);
                return this;
            }

            public Builder addAllInlierIdMatchPos(Iterable<? extends Integer> values) {
                copyOnWrite();
                ((MotionBoxState) this.instance).addAllInlierIdMatchPos(values);
                return this;
            }

            public Builder clearInlierIdMatchPos() {
                copyOnWrite();
                ((MotionBoxState) this.instance).clearInlierIdMatchPos();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public List<Integer> getInlierLengthList() {
                return Collections.unmodifiableList(((MotionBoxState) this.instance).getInlierLengthList());
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public int getInlierLengthCount() {
                return ((MotionBoxState) this.instance).getInlierLengthCount();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public int getInlierLength(int index) {
                return ((MotionBoxState) this.instance).getInlierLength(index);
            }

            public Builder setInlierLength(int index, int value) {
                copyOnWrite();
                ((MotionBoxState) this.instance).setInlierLength(index, value);
                return this;
            }

            public Builder addInlierLength(int value) {
                copyOnWrite();
                ((MotionBoxState) this.instance).addInlierLength(value);
                return this;
            }

            public Builder addAllInlierLength(Iterable<? extends Integer> values) {
                copyOnWrite();
                ((MotionBoxState) this.instance).addAllInlierLength(values);
                return this;
            }

            public Builder clearInlierLength() {
                copyOnWrite();
                ((MotionBoxState) this.instance).clearInlierLength();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public List<Integer> getOutlierIdsList() {
                return Collections.unmodifiableList(((MotionBoxState) this.instance).getOutlierIdsList());
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public int getOutlierIdsCount() {
                return ((MotionBoxState) this.instance).getOutlierIdsCount();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public int getOutlierIds(int index) {
                return ((MotionBoxState) this.instance).getOutlierIds(index);
            }

            public Builder setOutlierIds(int index, int value) {
                copyOnWrite();
                ((MotionBoxState) this.instance).setOutlierIds(index, value);
                return this;
            }

            public Builder addOutlierIds(int value) {
                copyOnWrite();
                ((MotionBoxState) this.instance).addOutlierIds(value);
                return this;
            }

            public Builder addAllOutlierIds(Iterable<? extends Integer> values) {
                copyOnWrite();
                ((MotionBoxState) this.instance).addAllOutlierIds(values);
                return this;
            }

            public Builder clearOutlierIds() {
                copyOnWrite();
                ((MotionBoxState) this.instance).clearOutlierIds();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public List<Integer> getOutlierIdMatchPosList() {
                return Collections.unmodifiableList(((MotionBoxState) this.instance).getOutlierIdMatchPosList());
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public int getOutlierIdMatchPosCount() {
                return ((MotionBoxState) this.instance).getOutlierIdMatchPosCount();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public int getOutlierIdMatchPos(int index) {
                return ((MotionBoxState) this.instance).getOutlierIdMatchPos(index);
            }

            public Builder setOutlierIdMatchPos(int index, int value) {
                copyOnWrite();
                ((MotionBoxState) this.instance).setOutlierIdMatchPos(index, value);
                return this;
            }

            public Builder addOutlierIdMatchPos(int value) {
                copyOnWrite();
                ((MotionBoxState) this.instance).addOutlierIdMatchPos(value);
                return this;
            }

            public Builder addAllOutlierIdMatchPos(Iterable<? extends Integer> values) {
                copyOnWrite();
                ((MotionBoxState) this.instance).addAllOutlierIdMatchPos(values);
                return this;
            }

            public Builder clearOutlierIdMatchPos() {
                copyOnWrite();
                ((MotionBoxState) this.instance).clearOutlierIdMatchPos();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public boolean hasTrackingConfidence() {
                return ((MotionBoxState) this.instance).hasTrackingConfidence();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public float getTrackingConfidence() {
                return ((MotionBoxState) this.instance).getTrackingConfidence();
            }

            public Builder setTrackingConfidence(float value) {
                copyOnWrite();
                ((MotionBoxState) this.instance).setTrackingConfidence(value);
                return this;
            }

            public Builder clearTrackingConfidence() {
                copyOnWrite();
                ((MotionBoxState) this.instance).clearTrackingConfidence();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public boolean hasInternal() {
                return ((MotionBoxState) this.instance).hasInternal();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxStateOrBuilder
            public MotionBoxInternalState getInternal() {
                return ((MotionBoxState) this.instance).getInternal();
            }

            public Builder setInternal(MotionBoxInternalState value) {
                copyOnWrite();
                ((MotionBoxState) this.instance).setInternal(value);
                return this;
            }

            public Builder setInternal(MotionBoxInternalState.Builder builderForValue) {
                copyOnWrite();
                ((MotionBoxState) this.instance).setInternal(builderForValue.build());
                return this;
            }

            public Builder mergeInternal(MotionBoxInternalState value) {
                copyOnWrite();
                ((MotionBoxState) this.instance).mergeInternal(value);
                return this;
            }

            public Builder clearInternal() {
                copyOnWrite();
                ((MotionBoxState) this.instance).clearInternal();
                return this;
            }
        }

        /* JADX INFO: Access modifiers changed from: protected */
        @Override // com.google.protobuf.GeneratedMessageLite
        public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
            switch (method) {
                case NEW_MUTABLE_INSTANCE:
                    return new MotionBoxState();
                case NEW_BUILDER:
                    return new Builder();
                case BUILD_MESSAGE_INFO:
                    Object[] objects = {"bitField0_", "posX_", "posY_", "width_", "height_", "scale_", "dx_", "dy_", "priorWeight_", "trackStatus_", TrackStatus.internalGetVerifier(), "spatialPriorGridSize_", "spatialPrior_", "spatialConfidence_", "priorDiff_", "motionDisparity_", "backgroundDiscrimination_", "kineticEnergy_", "inlierCenterX_", "inlierCenterY_", "inlierWidth_", "inlierHeight_", "inlierSum_", "inlierRatio_", "inlierIds_", "inlierLength_", "outlierIds_", "internal_", "rotation_", "inlierIdMatchPos_", "outlierIdMatchPos_", "trackingConfidence_", "quad_", "aspectRatio_", "pnpHomography_", "requestGrouping_"};
                    return newMessageInfo(DEFAULT_INSTANCE, "\u0001\"��\u0001\u0001%\"��\u0007��\u0001\u0001��\u0002\u0001\u0001\u0003\u0001\u0002\u0004\u0001\u0003\u0005\u0001\u0004\u0007\u0001\n\b\u0001\u000b\t\u0001\r\n\f\u000e\u000b\u0004\u000f\f$\r$\u000e\u0001\u0010\u000f\u0001\u0011\u0010\u0001\u0012\u0011\u0001\f\u0012\u0001\u0013\u0013\u0001\u0014\u0016\u0001\u0017\u0017\u0001\u0018\u0018\u0001\u0015\u0019\u0001\u0016\u001a+\u001b+\u001c+\u001d\t\u001a\u001e\u0001\u0005\u001f+ +!\u0001\u0019\"\t\u0006#\u0001\u0007$\t\t%\u0007\b", objects);
                case GET_DEFAULT_INSTANCE:
                    return DEFAULT_INSTANCE;
                case GET_PARSER:
                    Parser<MotionBoxState> parser = PARSER;
                    if (parser == null) {
                        synchronized (MotionBoxState.class) {
                            parser = PARSER;
                            if (parser == null) {
                                parser = new GeneratedMessageLite.DefaultInstanceBasedParser<>(DEFAULT_INSTANCE);
                                PARSER = parser;
                            }
                        }
                    }
                    return parser;
                case GET_MEMOIZED_IS_INITIALIZED:
                    return (byte) 1;
                case SET_MEMOIZED_IS_INITIALIZED:
                    return null;
                default:
                    throw new UnsupportedOperationException();
            }
        }

        static {
            MotionBoxState defaultInstance = new MotionBoxState();
            DEFAULT_INSTANCE = defaultInstance;
            GeneratedMessageLite.registerDefaultInstance(MotionBoxState.class, defaultInstance);
        }

        public static MotionBoxState getDefaultInstance() {
            return DEFAULT_INSTANCE;
        }

        public static Parser<MotionBoxState> parser() {
            return DEFAULT_INSTANCE.getParserForType();
        }
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/TrackingProto$MotionBoxInternalState.class */
    public static final class MotionBoxInternalState extends GeneratedMessageLite<MotionBoxInternalState, Builder> implements MotionBoxInternalStateOrBuilder {
        public static final int POS_X_FIELD_NUMBER = 1;
        public static final int POS_Y_FIELD_NUMBER = 2;
        public static final int DX_FIELD_NUMBER = 3;
        public static final int DY_FIELD_NUMBER = 4;
        public static final int CAMERA_DX_FIELD_NUMBER = 5;
        public static final int CAMERA_DY_FIELD_NUMBER = 6;
        public static final int TRACK_ID_FIELD_NUMBER = 7;
        public static final int INLIER_SCORE_FIELD_NUMBER = 8;
        private static final MotionBoxInternalState DEFAULT_INSTANCE;
        private static volatile Parser<MotionBoxInternalState> PARSER;
        private int posXMemoizedSerializedSize = -1;
        private int posYMemoizedSerializedSize = -1;
        private int dxMemoizedSerializedSize = -1;
        private int dyMemoizedSerializedSize = -1;
        private int cameraDxMemoizedSerializedSize = -1;
        private int cameraDyMemoizedSerializedSize = -1;
        private int trackIdMemoizedSerializedSize = -1;
        private int inlierScoreMemoizedSerializedSize = -1;
        private Internal.FloatList posX_ = emptyFloatList();
        private Internal.FloatList posY_ = emptyFloatList();
        private Internal.FloatList dx_ = emptyFloatList();
        private Internal.FloatList dy_ = emptyFloatList();
        private Internal.FloatList cameraDx_ = emptyFloatList();
        private Internal.FloatList cameraDy_ = emptyFloatList();
        private Internal.IntList trackId_ = emptyIntList();
        private Internal.FloatList inlierScore_ = emptyFloatList();

        private MotionBoxInternalState() {
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
        public List<Float> getPosXList() {
            return this.posX_;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
        public int getPosXCount() {
            return this.posX_.size();
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
        public float getPosX(int index) {
            return this.posX_.getFloat(index);
        }

        private void ensurePosXIsMutable() {
            if (!this.posX_.isModifiable()) {
                this.posX_ = GeneratedMessageLite.mutableCopy(this.posX_);
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setPosX(int index, float value) {
            ensurePosXIsMutable();
            this.posX_.setFloat(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addPosX(float value) {
            ensurePosXIsMutable();
            this.posX_.addFloat(value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addAllPosX(Iterable<? extends Float> values) {
            ensurePosXIsMutable();
            AbstractMessageLite.addAll((Iterable) values, (List) this.posX_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearPosX() {
            this.posX_ = emptyFloatList();
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
        public List<Float> getPosYList() {
            return this.posY_;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
        public int getPosYCount() {
            return this.posY_.size();
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
        public float getPosY(int index) {
            return this.posY_.getFloat(index);
        }

        private void ensurePosYIsMutable() {
            if (!this.posY_.isModifiable()) {
                this.posY_ = GeneratedMessageLite.mutableCopy(this.posY_);
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setPosY(int index, float value) {
            ensurePosYIsMutable();
            this.posY_.setFloat(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addPosY(float value) {
            ensurePosYIsMutable();
            this.posY_.addFloat(value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addAllPosY(Iterable<? extends Float> values) {
            ensurePosYIsMutable();
            AbstractMessageLite.addAll((Iterable) values, (List) this.posY_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearPosY() {
            this.posY_ = emptyFloatList();
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
        public List<Float> getDxList() {
            return this.dx_;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
        public int getDxCount() {
            return this.dx_.size();
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
        public float getDx(int index) {
            return this.dx_.getFloat(index);
        }

        private void ensureDxIsMutable() {
            if (!this.dx_.isModifiable()) {
                this.dx_ = GeneratedMessageLite.mutableCopy(this.dx_);
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setDx(int index, float value) {
            ensureDxIsMutable();
            this.dx_.setFloat(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addDx(float value) {
            ensureDxIsMutable();
            this.dx_.addFloat(value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addAllDx(Iterable<? extends Float> values) {
            ensureDxIsMutable();
            AbstractMessageLite.addAll((Iterable) values, (List) this.dx_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearDx() {
            this.dx_ = emptyFloatList();
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
        public List<Float> getDyList() {
            return this.dy_;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
        public int getDyCount() {
            return this.dy_.size();
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
        public float getDy(int index) {
            return this.dy_.getFloat(index);
        }

        private void ensureDyIsMutable() {
            if (!this.dy_.isModifiable()) {
                this.dy_ = GeneratedMessageLite.mutableCopy(this.dy_);
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setDy(int index, float value) {
            ensureDyIsMutable();
            this.dy_.setFloat(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addDy(float value) {
            ensureDyIsMutable();
            this.dy_.addFloat(value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addAllDy(Iterable<? extends Float> values) {
            ensureDyIsMutable();
            AbstractMessageLite.addAll((Iterable) values, (List) this.dy_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearDy() {
            this.dy_ = emptyFloatList();
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
        public List<Float> getCameraDxList() {
            return this.cameraDx_;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
        public int getCameraDxCount() {
            return this.cameraDx_.size();
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
        public float getCameraDx(int index) {
            return this.cameraDx_.getFloat(index);
        }

        private void ensureCameraDxIsMutable() {
            if (!this.cameraDx_.isModifiable()) {
                this.cameraDx_ = GeneratedMessageLite.mutableCopy(this.cameraDx_);
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setCameraDx(int index, float value) {
            ensureCameraDxIsMutable();
            this.cameraDx_.setFloat(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addCameraDx(float value) {
            ensureCameraDxIsMutable();
            this.cameraDx_.addFloat(value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addAllCameraDx(Iterable<? extends Float> values) {
            ensureCameraDxIsMutable();
            AbstractMessageLite.addAll((Iterable) values, (List) this.cameraDx_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearCameraDx() {
            this.cameraDx_ = emptyFloatList();
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
        public List<Float> getCameraDyList() {
            return this.cameraDy_;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
        public int getCameraDyCount() {
            return this.cameraDy_.size();
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
        public float getCameraDy(int index) {
            return this.cameraDy_.getFloat(index);
        }

        private void ensureCameraDyIsMutable() {
            if (!this.cameraDy_.isModifiable()) {
                this.cameraDy_ = GeneratedMessageLite.mutableCopy(this.cameraDy_);
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setCameraDy(int index, float value) {
            ensureCameraDyIsMutable();
            this.cameraDy_.setFloat(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addCameraDy(float value) {
            ensureCameraDyIsMutable();
            this.cameraDy_.addFloat(value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addAllCameraDy(Iterable<? extends Float> values) {
            ensureCameraDyIsMutable();
            AbstractMessageLite.addAll((Iterable) values, (List) this.cameraDy_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearCameraDy() {
            this.cameraDy_ = emptyFloatList();
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
        public List<Integer> getTrackIdList() {
            return this.trackId_;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
        public int getTrackIdCount() {
            return this.trackId_.size();
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
        public int getTrackId(int index) {
            return this.trackId_.getInt(index);
        }

        private void ensureTrackIdIsMutable() {
            if (!this.trackId_.isModifiable()) {
                this.trackId_ = GeneratedMessageLite.mutableCopy(this.trackId_);
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setTrackId(int index, int value) {
            ensureTrackIdIsMutable();
            this.trackId_.setInt(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addTrackId(int value) {
            ensureTrackIdIsMutable();
            this.trackId_.addInt(value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addAllTrackId(Iterable<? extends Integer> values) {
            ensureTrackIdIsMutable();
            AbstractMessageLite.addAll((Iterable) values, (List) this.trackId_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearTrackId() {
            this.trackId_ = emptyIntList();
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
        public List<Float> getInlierScoreList() {
            return this.inlierScore_;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
        public int getInlierScoreCount() {
            return this.inlierScore_.size();
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
        public float getInlierScore(int index) {
            return this.inlierScore_.getFloat(index);
        }

        private void ensureInlierScoreIsMutable() {
            if (!this.inlierScore_.isModifiable()) {
                this.inlierScore_ = GeneratedMessageLite.mutableCopy(this.inlierScore_);
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setInlierScore(int index, float value) {
            ensureInlierScoreIsMutable();
            this.inlierScore_.setFloat(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addInlierScore(float value) {
            ensureInlierScoreIsMutable();
            this.inlierScore_.addFloat(value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addAllInlierScore(Iterable<? extends Float> values) {
            ensureInlierScoreIsMutable();
            AbstractMessageLite.addAll((Iterable) values, (List) this.inlierScore_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearInlierScore() {
            this.inlierScore_ = emptyFloatList();
        }

        public static MotionBoxInternalState parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
            return (MotionBoxInternalState) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static MotionBoxInternalState parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (MotionBoxInternalState) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static MotionBoxInternalState parseFrom(ByteString data) throws InvalidProtocolBufferException {
            return (MotionBoxInternalState) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static MotionBoxInternalState parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (MotionBoxInternalState) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static MotionBoxInternalState parseFrom(byte[] data) throws InvalidProtocolBufferException {
            return (MotionBoxInternalState) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static MotionBoxInternalState parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (MotionBoxInternalState) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static MotionBoxInternalState parseFrom(InputStream input) throws IOException {
            return (MotionBoxInternalState) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static MotionBoxInternalState parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (MotionBoxInternalState) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static MotionBoxInternalState parseDelimitedFrom(InputStream input) throws IOException {
            return (MotionBoxInternalState) parseDelimitedFrom(DEFAULT_INSTANCE, input);
        }

        public static MotionBoxInternalState parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (MotionBoxInternalState) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static MotionBoxInternalState parseFrom(CodedInputStream input) throws IOException {
            return (MotionBoxInternalState) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static MotionBoxInternalState parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (MotionBoxInternalState) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Builder newBuilder() {
            return DEFAULT_INSTANCE.createBuilder();
        }

        public static Builder newBuilder(MotionBoxInternalState prototype) {
            return DEFAULT_INSTANCE.createBuilder(prototype);
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/TrackingProto$MotionBoxInternalState$Builder.class */
        public static final class Builder extends GeneratedMessageLite.Builder<MotionBoxInternalState, Builder> implements MotionBoxInternalStateOrBuilder {
            private Builder() {
                super(MotionBoxInternalState.DEFAULT_INSTANCE);
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
            public List<Float> getPosXList() {
                return Collections.unmodifiableList(((MotionBoxInternalState) this.instance).getPosXList());
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
            public int getPosXCount() {
                return ((MotionBoxInternalState) this.instance).getPosXCount();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
            public float getPosX(int index) {
                return ((MotionBoxInternalState) this.instance).getPosX(index);
            }

            public Builder setPosX(int index, float value) {
                copyOnWrite();
                ((MotionBoxInternalState) this.instance).setPosX(index, value);
                return this;
            }

            public Builder addPosX(float value) {
                copyOnWrite();
                ((MotionBoxInternalState) this.instance).addPosX(value);
                return this;
            }

            public Builder addAllPosX(Iterable<? extends Float> values) {
                copyOnWrite();
                ((MotionBoxInternalState) this.instance).addAllPosX(values);
                return this;
            }

            public Builder clearPosX() {
                copyOnWrite();
                ((MotionBoxInternalState) this.instance).clearPosX();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
            public List<Float> getPosYList() {
                return Collections.unmodifiableList(((MotionBoxInternalState) this.instance).getPosYList());
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
            public int getPosYCount() {
                return ((MotionBoxInternalState) this.instance).getPosYCount();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
            public float getPosY(int index) {
                return ((MotionBoxInternalState) this.instance).getPosY(index);
            }

            public Builder setPosY(int index, float value) {
                copyOnWrite();
                ((MotionBoxInternalState) this.instance).setPosY(index, value);
                return this;
            }

            public Builder addPosY(float value) {
                copyOnWrite();
                ((MotionBoxInternalState) this.instance).addPosY(value);
                return this;
            }

            public Builder addAllPosY(Iterable<? extends Float> values) {
                copyOnWrite();
                ((MotionBoxInternalState) this.instance).addAllPosY(values);
                return this;
            }

            public Builder clearPosY() {
                copyOnWrite();
                ((MotionBoxInternalState) this.instance).clearPosY();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
            public List<Float> getDxList() {
                return Collections.unmodifiableList(((MotionBoxInternalState) this.instance).getDxList());
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
            public int getDxCount() {
                return ((MotionBoxInternalState) this.instance).getDxCount();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
            public float getDx(int index) {
                return ((MotionBoxInternalState) this.instance).getDx(index);
            }

            public Builder setDx(int index, float value) {
                copyOnWrite();
                ((MotionBoxInternalState) this.instance).setDx(index, value);
                return this;
            }

            public Builder addDx(float value) {
                copyOnWrite();
                ((MotionBoxInternalState) this.instance).addDx(value);
                return this;
            }

            public Builder addAllDx(Iterable<? extends Float> values) {
                copyOnWrite();
                ((MotionBoxInternalState) this.instance).addAllDx(values);
                return this;
            }

            public Builder clearDx() {
                copyOnWrite();
                ((MotionBoxInternalState) this.instance).clearDx();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
            public List<Float> getDyList() {
                return Collections.unmodifiableList(((MotionBoxInternalState) this.instance).getDyList());
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
            public int getDyCount() {
                return ((MotionBoxInternalState) this.instance).getDyCount();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
            public float getDy(int index) {
                return ((MotionBoxInternalState) this.instance).getDy(index);
            }

            public Builder setDy(int index, float value) {
                copyOnWrite();
                ((MotionBoxInternalState) this.instance).setDy(index, value);
                return this;
            }

            public Builder addDy(float value) {
                copyOnWrite();
                ((MotionBoxInternalState) this.instance).addDy(value);
                return this;
            }

            public Builder addAllDy(Iterable<? extends Float> values) {
                copyOnWrite();
                ((MotionBoxInternalState) this.instance).addAllDy(values);
                return this;
            }

            public Builder clearDy() {
                copyOnWrite();
                ((MotionBoxInternalState) this.instance).clearDy();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
            public List<Float> getCameraDxList() {
                return Collections.unmodifiableList(((MotionBoxInternalState) this.instance).getCameraDxList());
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
            public int getCameraDxCount() {
                return ((MotionBoxInternalState) this.instance).getCameraDxCount();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
            public float getCameraDx(int index) {
                return ((MotionBoxInternalState) this.instance).getCameraDx(index);
            }

            public Builder setCameraDx(int index, float value) {
                copyOnWrite();
                ((MotionBoxInternalState) this.instance).setCameraDx(index, value);
                return this;
            }

            public Builder addCameraDx(float value) {
                copyOnWrite();
                ((MotionBoxInternalState) this.instance).addCameraDx(value);
                return this;
            }

            public Builder addAllCameraDx(Iterable<? extends Float> values) {
                copyOnWrite();
                ((MotionBoxInternalState) this.instance).addAllCameraDx(values);
                return this;
            }

            public Builder clearCameraDx() {
                copyOnWrite();
                ((MotionBoxInternalState) this.instance).clearCameraDx();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
            public List<Float> getCameraDyList() {
                return Collections.unmodifiableList(((MotionBoxInternalState) this.instance).getCameraDyList());
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
            public int getCameraDyCount() {
                return ((MotionBoxInternalState) this.instance).getCameraDyCount();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
            public float getCameraDy(int index) {
                return ((MotionBoxInternalState) this.instance).getCameraDy(index);
            }

            public Builder setCameraDy(int index, float value) {
                copyOnWrite();
                ((MotionBoxInternalState) this.instance).setCameraDy(index, value);
                return this;
            }

            public Builder addCameraDy(float value) {
                copyOnWrite();
                ((MotionBoxInternalState) this.instance).addCameraDy(value);
                return this;
            }

            public Builder addAllCameraDy(Iterable<? extends Float> values) {
                copyOnWrite();
                ((MotionBoxInternalState) this.instance).addAllCameraDy(values);
                return this;
            }

            public Builder clearCameraDy() {
                copyOnWrite();
                ((MotionBoxInternalState) this.instance).clearCameraDy();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
            public List<Integer> getTrackIdList() {
                return Collections.unmodifiableList(((MotionBoxInternalState) this.instance).getTrackIdList());
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
            public int getTrackIdCount() {
                return ((MotionBoxInternalState) this.instance).getTrackIdCount();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
            public int getTrackId(int index) {
                return ((MotionBoxInternalState) this.instance).getTrackId(index);
            }

            public Builder setTrackId(int index, int value) {
                copyOnWrite();
                ((MotionBoxInternalState) this.instance).setTrackId(index, value);
                return this;
            }

            public Builder addTrackId(int value) {
                copyOnWrite();
                ((MotionBoxInternalState) this.instance).addTrackId(value);
                return this;
            }

            public Builder addAllTrackId(Iterable<? extends Integer> values) {
                copyOnWrite();
                ((MotionBoxInternalState) this.instance).addAllTrackId(values);
                return this;
            }

            public Builder clearTrackId() {
                copyOnWrite();
                ((MotionBoxInternalState) this.instance).clearTrackId();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
            public List<Float> getInlierScoreList() {
                return Collections.unmodifiableList(((MotionBoxInternalState) this.instance).getInlierScoreList());
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
            public int getInlierScoreCount() {
                return ((MotionBoxInternalState) this.instance).getInlierScoreCount();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.MotionBoxInternalStateOrBuilder
            public float getInlierScore(int index) {
                return ((MotionBoxInternalState) this.instance).getInlierScore(index);
            }

            public Builder setInlierScore(int index, float value) {
                copyOnWrite();
                ((MotionBoxInternalState) this.instance).setInlierScore(index, value);
                return this;
            }

            public Builder addInlierScore(float value) {
                copyOnWrite();
                ((MotionBoxInternalState) this.instance).addInlierScore(value);
                return this;
            }

            public Builder addAllInlierScore(Iterable<? extends Float> values) {
                copyOnWrite();
                ((MotionBoxInternalState) this.instance).addAllInlierScore(values);
                return this;
            }

            public Builder clearInlierScore() {
                copyOnWrite();
                ((MotionBoxInternalState) this.instance).clearInlierScore();
                return this;
            }
        }

        /* JADX INFO: Access modifiers changed from: protected */
        @Override // com.google.protobuf.GeneratedMessageLite
        public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
            switch (method) {
                case NEW_MUTABLE_INSTANCE:
                    return new MotionBoxInternalState();
                case NEW_BUILDER:
                    return new Builder();
                case BUILD_MESSAGE_INFO:
                    Object[] objects = {"posX_", "posY_", "dx_", "dy_", "cameraDx_", "cameraDy_", "trackId_", "inlierScore_"};
                    return newMessageInfo(DEFAULT_INSTANCE, "\u0001\b����\u0001\b\b��\b��\u0001$\u0002$\u0003$\u0004$\u0005$\u0006$\u0007'\b$", objects);
                case GET_DEFAULT_INSTANCE:
                    return DEFAULT_INSTANCE;
                case GET_PARSER:
                    Parser<MotionBoxInternalState> parser = PARSER;
                    if (parser == null) {
                        synchronized (MotionBoxInternalState.class) {
                            parser = PARSER;
                            if (parser == null) {
                                parser = new GeneratedMessageLite.DefaultInstanceBasedParser<>(DEFAULT_INSTANCE);
                                PARSER = parser;
                            }
                        }
                    }
                    return parser;
                case GET_MEMOIZED_IS_INITIALIZED:
                    return (byte) 1;
                case SET_MEMOIZED_IS_INITIALIZED:
                    return null;
                default:
                    throw new UnsupportedOperationException();
            }
        }

        static {
            MotionBoxInternalState defaultInstance = new MotionBoxInternalState();
            DEFAULT_INSTANCE = defaultInstance;
            GeneratedMessageLite.registerDefaultInstance(MotionBoxInternalState.class, defaultInstance);
        }

        public static MotionBoxInternalState getDefaultInstance() {
            return DEFAULT_INSTANCE;
        }

        public static Parser<MotionBoxInternalState> parser() {
            return DEFAULT_INSTANCE.getParserForType();
        }
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/TrackingProto$TrackStepOptions.class */
    public static final class TrackStepOptions extends GeneratedMessageLite<TrackStepOptions, Builder> implements TrackStepOptionsOrBuilder {
        private int bitField0_;
        private int bitField1_;
        public static final int TRACKING_DEGREES_FIELD_NUMBER = 28;
        private int trackingDegrees_;
        public static final int TRACK_OBJECT_AND_CAMERA_FIELD_NUMBER = 32;
        private boolean trackObjectAndCamera_;
        public static final int IRLS_ITERATIONS_FIELD_NUMBER = 1;
        public static final int SPATIAL_SIGMA_FIELD_NUMBER = 2;
        public static final int MIN_MOTION_SIGMA_FIELD_NUMBER = 3;
        public static final int RELATIVE_MOTION_SIGMA_FIELD_NUMBER = 4;
        public static final int MOTION_DISPARITY_LOW_LEVEL_FIELD_NUMBER = 6;
        public static final int MOTION_DISPARITY_HIGH_LEVEL_FIELD_NUMBER = 7;
        public static final int DISPARITY_DECAY_FIELD_NUMBER = 8;
        public static final int MOTION_PRIOR_WEIGHT_FIELD_NUMBER = 9;
        public static final int BACKGROUND_DISCRIMINATION_LOW_LEVEL_FIELD_NUMBER = 10;
        public static final int BACKGROUND_DISCRIMINATION_HIGH_LEVEL_FIELD_NUMBER = 11;
        public static final int INLIER_CENTER_RELATIVE_DISTANCE_FIELD_NUMBER = 12;
        public static final int INLIER_SPRING_FORCE_FIELD_NUMBER = 13;
        public static final int KINETIC_CENTER_RELATIVE_DISTANCE_FIELD_NUMBER = 14;
        public static final int KINETIC_SPRING_FORCE_FIELD_NUMBER = 15;
        public static final int KINETIC_SPRING_FORCE_MIN_KINETIC_ENERGY_FIELD_NUMBER = 21;
        public static final int VELOCITY_UPDATE_WEIGHT_FIELD_NUMBER = 16;
        public static final int MAX_TRACK_FAILURES_FIELD_NUMBER = 17;
        public static final int EXPANSION_SIZE_FIELD_NUMBER = 18;
        public static final int INLIER_LOW_WEIGHT_FIELD_NUMBER = 19;
        public static final int INLIER_HIGH_WEIGHT_FIELD_NUMBER = 20;
        public static final int KINETIC_ENERGY_DECAY_FIELD_NUMBER = 22;
        public static final int PRIOR_WEIGHT_INCREASE_FIELD_NUMBER = 23;
        public static final int LOW_KINETIC_ENERGY_FIELD_NUMBER = 24;
        public static final int HIGH_KINETIC_ENERGY_FIELD_NUMBER = 25;
        public static final int RETURN_INTERNAL_STATE_FIELD_NUMBER = 26;
        private boolean returnInternalState_;
        public static final int USE_POST_ESTIMATION_WEIGHTS_FOR_STATE_FIELD_NUMBER = 29;
        public static final int COMPUTE_SPATIAL_PRIOR_FIELD_NUMBER = 27;
        private boolean computeSpatialPrior_;
        public static final int IRLS_INITIALIZATION_FIELD_NUMBER = 30;
        private IrlsInitialization irlsInitialization_;
        public static final int STATIC_MOTION_TEMPORAL_RATIO_FIELD_NUMBER = 33;
        public static final int CANCEL_TRACKING_WITH_OCCLUSION_OPTIONS_FIELD_NUMBER = 34;
        private CancelTrackingWithOcclusionOptions cancelTrackingWithOcclusionOptions_;
        public static final int OBJECT_SIMILARITY_MIN_CONTD_INLIERS_FIELD_NUMBER = 35;
        public static final int BOX_SIMILARITY_MAX_SCALE_FIELD_NUMBER = 36;
        public static final int BOX_SIMILARITY_MAX_ROTATION_FIELD_NUMBER = 37;
        public static final int QUAD_HOMOGRAPHY_MAX_SCALE_FIELD_NUMBER = 38;
        public static final int QUAD_HOMOGRAPHY_MAX_ROTATION_FIELD_NUMBER = 39;
        public static final int CAMERA_INTRINSICS_FIELD_NUMBER = 40;
        private CameraIntrinsics cameraIntrinsics_;
        public static final int FORCED_PNP_TRACKING_FIELD_NUMBER = 41;
        private boolean forcedPnpTracking_;
        private static final TrackStepOptions DEFAULT_INSTANCE;
        private static volatile Parser<TrackStepOptions> PARSER;
        private int irlsIterations_ = 5;
        private float spatialSigma_ = 0.15f;
        private float minMotionSigma_ = 0.002f;
        private float relativeMotionSigma_ = 0.3f;
        private float motionDisparityLowLevel_ = 0.008f;
        private float motionDisparityHighLevel_ = 0.016f;
        private float disparityDecay_ = 0.8f;
        private float motionPriorWeight_ = 0.2f;
        private float backgroundDiscriminationLowLevel_ = 0.004f;
        private float backgroundDiscriminationHighLevel_ = 0.008f;
        private float inlierCenterRelativeDistance_ = 0.1f;
        private float inlierSpringForce_ = 0.3f;
        private float kineticCenterRelativeDistance_ = 0.4f;
        private float kineticSpringForce_ = 0.5f;
        private float kineticSpringForceMinKineticEnergy_ = 0.003f;
        private float velocityUpdateWeight_ = 0.7f;
        private int maxTrackFailures_ = 10;
        private float expansionSize_ = 0.05f;
        private float inlierLowWeight_ = 250.0f;
        private float inlierHighWeight_ = 500.0f;
        private float kineticEnergyDecay_ = 0.98f;
        private float priorWeightIncrease_ = 0.2f;
        private float lowKineticEnergy_ = 0.001f;
        private float highKineticEnergy_ = 0.004f;
        private boolean usePostEstimationWeightsForState_ = true;
        private float staticMotionTemporalRatio_ = 0.003f;
        private int objectSimilarityMinContdInliers_ = 30;
        private float boxSimilarityMaxScale_ = 1.05f;
        private float boxSimilarityMaxRotation_ = 0.2f;
        private float quadHomographyMaxScale_ = 1.2f;
        private float quadHomographyMaxRotation_ = 0.3f;

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/TrackingProto$TrackStepOptions$CameraIntrinsicsOrBuilder.class */
        public interface CameraIntrinsicsOrBuilder extends MessageLiteOrBuilder {
            boolean hasFx();

            float getFx();

            boolean hasFy();

            float getFy();

            boolean hasCx();

            float getCx();

            boolean hasCy();

            float getCy();

            boolean hasK0();

            float getK0();

            boolean hasK1();

            float getK1();

            boolean hasK2();

            float getK2();

            boolean hasW();

            int getW();

            boolean hasH();

            int getH();
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/TrackingProto$TrackStepOptions$CancelTrackingWithOcclusionOptionsOrBuilder.class */
        public interface CancelTrackingWithOcclusionOptionsOrBuilder extends MessageLiteOrBuilder {
            boolean hasActivated();

            boolean getActivated();

            boolean hasMinMotionContinuity();

            float getMinMotionContinuity();

            boolean hasMinInlierRatio();

            float getMinInlierRatio();
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/TrackingProto$TrackStepOptions$IrlsInitializationOrBuilder.class */
        public interface IrlsInitializationOrBuilder extends MessageLiteOrBuilder {
            boolean hasActivated();

            boolean getActivated();

            boolean hasRounds();

            int getRounds();

            boolean hasCutoff();

            float getCutoff();
        }

        private TrackStepOptions() {
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/TrackingProto$TrackStepOptions$TrackingDegrees.class */
        public enum TrackingDegrees implements Internal.EnumLite {
            TRACKING_DEGREE_TRANSLATION(0),
            TRACKING_DEGREE_CAMERA_SCALE(1),
            TRACKING_DEGREE_CAMERA_ROTATION(2),
            TRACKING_DEGREE_CAMERA_ROTATION_SCALE(3),
            TRACKING_DEGREE_CAMERA_PERSPECTIVE(4),
            TRACKING_DEGREE_OBJECT_SCALE(5),
            TRACKING_DEGREE_OBJECT_ROTATION(6),
            TRACKING_DEGREE_OBJECT_ROTATION_SCALE(7),
            TRACKING_DEGREE_OBJECT_PERSPECTIVE(8);
            
            public static final int TRACKING_DEGREE_TRANSLATION_VALUE = 0;
            public static final int TRACKING_DEGREE_CAMERA_SCALE_VALUE = 1;
            public static final int TRACKING_DEGREE_CAMERA_ROTATION_VALUE = 2;
            public static final int TRACKING_DEGREE_CAMERA_ROTATION_SCALE_VALUE = 3;
            public static final int TRACKING_DEGREE_CAMERA_PERSPECTIVE_VALUE = 4;
            public static final int TRACKING_DEGREE_OBJECT_SCALE_VALUE = 5;
            public static final int TRACKING_DEGREE_OBJECT_ROTATION_VALUE = 6;
            public static final int TRACKING_DEGREE_OBJECT_ROTATION_SCALE_VALUE = 7;
            public static final int TRACKING_DEGREE_OBJECT_PERSPECTIVE_VALUE = 8;
            private static final Internal.EnumLiteMap<TrackingDegrees> internalValueMap = new Internal.EnumLiteMap<TrackingDegrees>() { // from class: com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.TrackingDegrees.1
                /* JADX DEBUG: Method merged with bridge method */
                /* JADX WARN: Can't rename method to resolve collision */
                @Override // com.google.protobuf.Internal.EnumLiteMap
                public TrackingDegrees findValueByNumber(int number) {
                    return TrackingDegrees.forNumber(number);
                }
            };
            private final int value;

            @Override // com.google.protobuf.Internal.EnumLite
            public final int getNumber() {
                return this.value;
            }

            @Deprecated
            public static TrackingDegrees valueOf(int value) {
                return forNumber(value);
            }

            public static TrackingDegrees forNumber(int value) {
                switch (value) {
                    case 0:
                        return TRACKING_DEGREE_TRANSLATION;
                    case 1:
                        return TRACKING_DEGREE_CAMERA_SCALE;
                    case 2:
                        return TRACKING_DEGREE_CAMERA_ROTATION;
                    case 3:
                        return TRACKING_DEGREE_CAMERA_ROTATION_SCALE;
                    case 4:
                        return TRACKING_DEGREE_CAMERA_PERSPECTIVE;
                    case 5:
                        return TRACKING_DEGREE_OBJECT_SCALE;
                    case 6:
                        return TRACKING_DEGREE_OBJECT_ROTATION;
                    case 7:
                        return TRACKING_DEGREE_OBJECT_ROTATION_SCALE;
                    case 8:
                        return TRACKING_DEGREE_OBJECT_PERSPECTIVE;
                    default:
                        return null;
                }
            }

            public static Internal.EnumLiteMap<TrackingDegrees> internalGetValueMap() {
                return internalValueMap;
            }

            public static Internal.EnumVerifier internalGetVerifier() {
                return TrackingDegreesVerifier.INSTANCE;
            }

            /* JADX INFO: Access modifiers changed from: private */
            /* JADX WARN: Classes with same name are omitted:
              classes2.dex
             */
            /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/TrackingProto$TrackStepOptions$TrackingDegrees$TrackingDegreesVerifier.class */
            public static final class TrackingDegreesVerifier implements Internal.EnumVerifier {
                static final Internal.EnumVerifier INSTANCE = new TrackingDegreesVerifier();

                private TrackingDegreesVerifier() {
                }

                @Override // com.google.protobuf.Internal.EnumVerifier
                public boolean isInRange(int number) {
                    return TrackingDegrees.forNumber(number) != null;
                }
            }

            TrackingDegrees(int value) {
                this.value = value;
            }
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/TrackingProto$TrackStepOptions$IrlsInitialization.class */
        public static final class IrlsInitialization extends GeneratedMessageLite<IrlsInitialization, Builder> implements IrlsInitializationOrBuilder {
            private int bitField0_;
            public static final int ACTIVATED_FIELD_NUMBER = 1;
            private boolean activated_;
            public static final int ROUNDS_FIELD_NUMBER = 2;
            public static final int CUTOFF_FIELD_NUMBER = 3;
            private static final IrlsInitialization DEFAULT_INSTANCE;
            private static volatile Parser<IrlsInitialization> PARSER;
            private int rounds_ = 50;
            private float cutoff_ = 0.005f;

            private IrlsInitialization() {
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.IrlsInitializationOrBuilder
            public boolean hasActivated() {
                return (this.bitField0_ & 1) != 0;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.IrlsInitializationOrBuilder
            public boolean getActivated() {
                return this.activated_;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setActivated(boolean value) {
                this.bitField0_ |= 1;
                this.activated_ = value;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void clearActivated() {
                this.bitField0_ &= -2;
                this.activated_ = false;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.IrlsInitializationOrBuilder
            public boolean hasRounds() {
                return (this.bitField0_ & 2) != 0;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.IrlsInitializationOrBuilder
            public int getRounds() {
                return this.rounds_;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setRounds(int value) {
                this.bitField0_ |= 2;
                this.rounds_ = value;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void clearRounds() {
                this.bitField0_ &= -3;
                this.rounds_ = 50;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.IrlsInitializationOrBuilder
            public boolean hasCutoff() {
                return (this.bitField0_ & 4) != 0;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.IrlsInitializationOrBuilder
            public float getCutoff() {
                return this.cutoff_;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setCutoff(float value) {
                this.bitField0_ |= 4;
                this.cutoff_ = value;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void clearCutoff() {
                this.bitField0_ &= -5;
                this.cutoff_ = 0.005f;
            }

            public static IrlsInitialization parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
                return (IrlsInitialization) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
            }

            public static IrlsInitialization parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
                return (IrlsInitialization) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
            }

            public static IrlsInitialization parseFrom(ByteString data) throws InvalidProtocolBufferException {
                return (IrlsInitialization) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
            }

            public static IrlsInitialization parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
                return (IrlsInitialization) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
            }

            public static IrlsInitialization parseFrom(byte[] data) throws InvalidProtocolBufferException {
                return (IrlsInitialization) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
            }

            public static IrlsInitialization parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
                return (IrlsInitialization) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
            }

            public static IrlsInitialization parseFrom(InputStream input) throws IOException {
                return (IrlsInitialization) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
            }

            public static IrlsInitialization parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
                return (IrlsInitialization) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
            }

            public static IrlsInitialization parseDelimitedFrom(InputStream input) throws IOException {
                return (IrlsInitialization) parseDelimitedFrom(DEFAULT_INSTANCE, input);
            }

            public static IrlsInitialization parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
                return (IrlsInitialization) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
            }

            public static IrlsInitialization parseFrom(CodedInputStream input) throws IOException {
                return (IrlsInitialization) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
            }

            public static IrlsInitialization parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
                return (IrlsInitialization) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
            }

            public static Builder newBuilder() {
                return DEFAULT_INSTANCE.createBuilder();
            }

            public static Builder newBuilder(IrlsInitialization prototype) {
                return DEFAULT_INSTANCE.createBuilder(prototype);
            }

            /* JADX WARN: Classes with same name are omitted:
              classes2.dex
             */
            /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/TrackingProto$TrackStepOptions$IrlsInitialization$Builder.class */
            public static final class Builder extends GeneratedMessageLite.Builder<IrlsInitialization, Builder> implements IrlsInitializationOrBuilder {
                private Builder() {
                    super(IrlsInitialization.DEFAULT_INSTANCE);
                }

                @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.IrlsInitializationOrBuilder
                public boolean hasActivated() {
                    return ((IrlsInitialization) this.instance).hasActivated();
                }

                @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.IrlsInitializationOrBuilder
                public boolean getActivated() {
                    return ((IrlsInitialization) this.instance).getActivated();
                }

                public Builder setActivated(boolean value) {
                    copyOnWrite();
                    ((IrlsInitialization) this.instance).setActivated(value);
                    return this;
                }

                public Builder clearActivated() {
                    copyOnWrite();
                    ((IrlsInitialization) this.instance).clearActivated();
                    return this;
                }

                @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.IrlsInitializationOrBuilder
                public boolean hasRounds() {
                    return ((IrlsInitialization) this.instance).hasRounds();
                }

                @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.IrlsInitializationOrBuilder
                public int getRounds() {
                    return ((IrlsInitialization) this.instance).getRounds();
                }

                public Builder setRounds(int value) {
                    copyOnWrite();
                    ((IrlsInitialization) this.instance).setRounds(value);
                    return this;
                }

                public Builder clearRounds() {
                    copyOnWrite();
                    ((IrlsInitialization) this.instance).clearRounds();
                    return this;
                }

                @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.IrlsInitializationOrBuilder
                public boolean hasCutoff() {
                    return ((IrlsInitialization) this.instance).hasCutoff();
                }

                @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.IrlsInitializationOrBuilder
                public float getCutoff() {
                    return ((IrlsInitialization) this.instance).getCutoff();
                }

                public Builder setCutoff(float value) {
                    copyOnWrite();
                    ((IrlsInitialization) this.instance).setCutoff(value);
                    return this;
                }

                public Builder clearCutoff() {
                    copyOnWrite();
                    ((IrlsInitialization) this.instance).clearCutoff();
                    return this;
                }
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.google.protobuf.GeneratedMessageLite
            public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
                switch (method) {
                    case NEW_MUTABLE_INSTANCE:
                        return new IrlsInitialization();
                    case NEW_BUILDER:
                        return new Builder();
                    case BUILD_MESSAGE_INFO:
                        Object[] objects = {"bitField0_", "activated_", "rounds_", "cutoff_"};
                        return newMessageInfo(DEFAULT_INSTANCE, "\u0001\u0003��\u0001\u0001\u0003\u0003������\u0001\u0007��\u0002\u0004\u0001\u0003\u0001\u0002", objects);
                    case GET_DEFAULT_INSTANCE:
                        return DEFAULT_INSTANCE;
                    case GET_PARSER:
                        Parser<IrlsInitialization> parser = PARSER;
                        if (parser == null) {
                            synchronized (IrlsInitialization.class) {
                                parser = PARSER;
                                if (parser == null) {
                                    parser = new GeneratedMessageLite.DefaultInstanceBasedParser<>(DEFAULT_INSTANCE);
                                    PARSER = parser;
                                }
                            }
                        }
                        return parser;
                    case GET_MEMOIZED_IS_INITIALIZED:
                        return (byte) 1;
                    case SET_MEMOIZED_IS_INITIALIZED:
                        return null;
                    default:
                        throw new UnsupportedOperationException();
                }
            }

            static {
                IrlsInitialization defaultInstance = new IrlsInitialization();
                DEFAULT_INSTANCE = defaultInstance;
                GeneratedMessageLite.registerDefaultInstance(IrlsInitialization.class, defaultInstance);
            }

            public static IrlsInitialization getDefaultInstance() {
                return DEFAULT_INSTANCE;
            }

            public static Parser<IrlsInitialization> parser() {
                return DEFAULT_INSTANCE.getParserForType();
            }
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/TrackingProto$TrackStepOptions$CancelTrackingWithOcclusionOptions.class */
        public static final class CancelTrackingWithOcclusionOptions extends GeneratedMessageLite<CancelTrackingWithOcclusionOptions, Builder> implements CancelTrackingWithOcclusionOptionsOrBuilder {
            private int bitField0_;
            public static final int ACTIVATED_FIELD_NUMBER = 1;
            private boolean activated_;
            public static final int MIN_MOTION_CONTINUITY_FIELD_NUMBER = 2;
            public static final int MIN_INLIER_RATIO_FIELD_NUMBER = 3;
            private static final CancelTrackingWithOcclusionOptions DEFAULT_INSTANCE;
            private static volatile Parser<CancelTrackingWithOcclusionOptions> PARSER;
            private float minMotionContinuity_ = 0.4f;
            private float minInlierRatio_ = 0.1f;

            private CancelTrackingWithOcclusionOptions() {
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CancelTrackingWithOcclusionOptionsOrBuilder
            public boolean hasActivated() {
                return (this.bitField0_ & 1) != 0;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CancelTrackingWithOcclusionOptionsOrBuilder
            public boolean getActivated() {
                return this.activated_;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setActivated(boolean value) {
                this.bitField0_ |= 1;
                this.activated_ = value;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void clearActivated() {
                this.bitField0_ &= -2;
                this.activated_ = false;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CancelTrackingWithOcclusionOptionsOrBuilder
            public boolean hasMinMotionContinuity() {
                return (this.bitField0_ & 2) != 0;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CancelTrackingWithOcclusionOptionsOrBuilder
            public float getMinMotionContinuity() {
                return this.minMotionContinuity_;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setMinMotionContinuity(float value) {
                this.bitField0_ |= 2;
                this.minMotionContinuity_ = value;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void clearMinMotionContinuity() {
                this.bitField0_ &= -3;
                this.minMotionContinuity_ = 0.4f;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CancelTrackingWithOcclusionOptionsOrBuilder
            public boolean hasMinInlierRatio() {
                return (this.bitField0_ & 4) != 0;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CancelTrackingWithOcclusionOptionsOrBuilder
            public float getMinInlierRatio() {
                return this.minInlierRatio_;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setMinInlierRatio(float value) {
                this.bitField0_ |= 4;
                this.minInlierRatio_ = value;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void clearMinInlierRatio() {
                this.bitField0_ &= -5;
                this.minInlierRatio_ = 0.1f;
            }

            public static CancelTrackingWithOcclusionOptions parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
                return (CancelTrackingWithOcclusionOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
            }

            public static CancelTrackingWithOcclusionOptions parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
                return (CancelTrackingWithOcclusionOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
            }

            public static CancelTrackingWithOcclusionOptions parseFrom(ByteString data) throws InvalidProtocolBufferException {
                return (CancelTrackingWithOcclusionOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
            }

            public static CancelTrackingWithOcclusionOptions parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
                return (CancelTrackingWithOcclusionOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
            }

            public static CancelTrackingWithOcclusionOptions parseFrom(byte[] data) throws InvalidProtocolBufferException {
                return (CancelTrackingWithOcclusionOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
            }

            public static CancelTrackingWithOcclusionOptions parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
                return (CancelTrackingWithOcclusionOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
            }

            public static CancelTrackingWithOcclusionOptions parseFrom(InputStream input) throws IOException {
                return (CancelTrackingWithOcclusionOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
            }

            public static CancelTrackingWithOcclusionOptions parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
                return (CancelTrackingWithOcclusionOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
            }

            public static CancelTrackingWithOcclusionOptions parseDelimitedFrom(InputStream input) throws IOException {
                return (CancelTrackingWithOcclusionOptions) parseDelimitedFrom(DEFAULT_INSTANCE, input);
            }

            public static CancelTrackingWithOcclusionOptions parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
                return (CancelTrackingWithOcclusionOptions) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
            }

            public static CancelTrackingWithOcclusionOptions parseFrom(CodedInputStream input) throws IOException {
                return (CancelTrackingWithOcclusionOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
            }

            public static CancelTrackingWithOcclusionOptions parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
                return (CancelTrackingWithOcclusionOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
            }

            public static Builder newBuilder() {
                return DEFAULT_INSTANCE.createBuilder();
            }

            public static Builder newBuilder(CancelTrackingWithOcclusionOptions prototype) {
                return DEFAULT_INSTANCE.createBuilder(prototype);
            }

            /* JADX WARN: Classes with same name are omitted:
              classes2.dex
             */
            /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/TrackingProto$TrackStepOptions$CancelTrackingWithOcclusionOptions$Builder.class */
            public static final class Builder extends GeneratedMessageLite.Builder<CancelTrackingWithOcclusionOptions, Builder> implements CancelTrackingWithOcclusionOptionsOrBuilder {
                private Builder() {
                    super(CancelTrackingWithOcclusionOptions.DEFAULT_INSTANCE);
                }

                @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CancelTrackingWithOcclusionOptionsOrBuilder
                public boolean hasActivated() {
                    return ((CancelTrackingWithOcclusionOptions) this.instance).hasActivated();
                }

                @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CancelTrackingWithOcclusionOptionsOrBuilder
                public boolean getActivated() {
                    return ((CancelTrackingWithOcclusionOptions) this.instance).getActivated();
                }

                public Builder setActivated(boolean value) {
                    copyOnWrite();
                    ((CancelTrackingWithOcclusionOptions) this.instance).setActivated(value);
                    return this;
                }

                public Builder clearActivated() {
                    copyOnWrite();
                    ((CancelTrackingWithOcclusionOptions) this.instance).clearActivated();
                    return this;
                }

                @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CancelTrackingWithOcclusionOptionsOrBuilder
                public boolean hasMinMotionContinuity() {
                    return ((CancelTrackingWithOcclusionOptions) this.instance).hasMinMotionContinuity();
                }

                @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CancelTrackingWithOcclusionOptionsOrBuilder
                public float getMinMotionContinuity() {
                    return ((CancelTrackingWithOcclusionOptions) this.instance).getMinMotionContinuity();
                }

                public Builder setMinMotionContinuity(float value) {
                    copyOnWrite();
                    ((CancelTrackingWithOcclusionOptions) this.instance).setMinMotionContinuity(value);
                    return this;
                }

                public Builder clearMinMotionContinuity() {
                    copyOnWrite();
                    ((CancelTrackingWithOcclusionOptions) this.instance).clearMinMotionContinuity();
                    return this;
                }

                @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CancelTrackingWithOcclusionOptionsOrBuilder
                public boolean hasMinInlierRatio() {
                    return ((CancelTrackingWithOcclusionOptions) this.instance).hasMinInlierRatio();
                }

                @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CancelTrackingWithOcclusionOptionsOrBuilder
                public float getMinInlierRatio() {
                    return ((CancelTrackingWithOcclusionOptions) this.instance).getMinInlierRatio();
                }

                public Builder setMinInlierRatio(float value) {
                    copyOnWrite();
                    ((CancelTrackingWithOcclusionOptions) this.instance).setMinInlierRatio(value);
                    return this;
                }

                public Builder clearMinInlierRatio() {
                    copyOnWrite();
                    ((CancelTrackingWithOcclusionOptions) this.instance).clearMinInlierRatio();
                    return this;
                }
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.google.protobuf.GeneratedMessageLite
            public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
                switch (method) {
                    case NEW_MUTABLE_INSTANCE:
                        return new CancelTrackingWithOcclusionOptions();
                    case NEW_BUILDER:
                        return new Builder();
                    case BUILD_MESSAGE_INFO:
                        Object[] objects = {"bitField0_", "activated_", "minMotionContinuity_", "minInlierRatio_"};
                        return newMessageInfo(DEFAULT_INSTANCE, "\u0001\u0003��\u0001\u0001\u0003\u0003������\u0001\u0007��\u0002\u0001\u0001\u0003\u0001\u0002", objects);
                    case GET_DEFAULT_INSTANCE:
                        return DEFAULT_INSTANCE;
                    case GET_PARSER:
                        Parser<CancelTrackingWithOcclusionOptions> parser = PARSER;
                        if (parser == null) {
                            synchronized (CancelTrackingWithOcclusionOptions.class) {
                                parser = PARSER;
                                if (parser == null) {
                                    parser = new GeneratedMessageLite.DefaultInstanceBasedParser<>(DEFAULT_INSTANCE);
                                    PARSER = parser;
                                }
                            }
                        }
                        return parser;
                    case GET_MEMOIZED_IS_INITIALIZED:
                        return (byte) 1;
                    case SET_MEMOIZED_IS_INITIALIZED:
                        return null;
                    default:
                        throw new UnsupportedOperationException();
                }
            }

            static {
                CancelTrackingWithOcclusionOptions defaultInstance = new CancelTrackingWithOcclusionOptions();
                DEFAULT_INSTANCE = defaultInstance;
                GeneratedMessageLite.registerDefaultInstance(CancelTrackingWithOcclusionOptions.class, defaultInstance);
            }

            public static CancelTrackingWithOcclusionOptions getDefaultInstance() {
                return DEFAULT_INSTANCE;
            }

            public static Parser<CancelTrackingWithOcclusionOptions> parser() {
                return DEFAULT_INSTANCE.getParserForType();
            }
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/TrackingProto$TrackStepOptions$CameraIntrinsics.class */
        public static final class CameraIntrinsics extends GeneratedMessageLite<CameraIntrinsics, Builder> implements CameraIntrinsicsOrBuilder {
            private int bitField0_;
            public static final int FX_FIELD_NUMBER = 1;
            private float fx_;
            public static final int FY_FIELD_NUMBER = 2;
            private float fy_;
            public static final int CX_FIELD_NUMBER = 3;
            private float cx_;
            public static final int CY_FIELD_NUMBER = 4;
            private float cy_;
            public static final int K0_FIELD_NUMBER = 5;
            private float k0_;
            public static final int K1_FIELD_NUMBER = 6;
            private float k1_;
            public static final int K2_FIELD_NUMBER = 7;
            private float k2_;
            public static final int W_FIELD_NUMBER = 8;
            private int w_;
            public static final int H_FIELD_NUMBER = 9;
            private int h_;
            private static final CameraIntrinsics DEFAULT_INSTANCE;
            private static volatile Parser<CameraIntrinsics> PARSER;

            private CameraIntrinsics() {
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CameraIntrinsicsOrBuilder
            public boolean hasFx() {
                return (this.bitField0_ & 1) != 0;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CameraIntrinsicsOrBuilder
            public float getFx() {
                return this.fx_;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setFx(float value) {
                this.bitField0_ |= 1;
                this.fx_ = value;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void clearFx() {
                this.bitField0_ &= -2;
                this.fx_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CameraIntrinsicsOrBuilder
            public boolean hasFy() {
                return (this.bitField0_ & 2) != 0;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CameraIntrinsicsOrBuilder
            public float getFy() {
                return this.fy_;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setFy(float value) {
                this.bitField0_ |= 2;
                this.fy_ = value;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void clearFy() {
                this.bitField0_ &= -3;
                this.fy_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CameraIntrinsicsOrBuilder
            public boolean hasCx() {
                return (this.bitField0_ & 4) != 0;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CameraIntrinsicsOrBuilder
            public float getCx() {
                return this.cx_;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setCx(float value) {
                this.bitField0_ |= 4;
                this.cx_ = value;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void clearCx() {
                this.bitField0_ &= -5;
                this.cx_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CameraIntrinsicsOrBuilder
            public boolean hasCy() {
                return (this.bitField0_ & 8) != 0;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CameraIntrinsicsOrBuilder
            public float getCy() {
                return this.cy_;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setCy(float value) {
                this.bitField0_ |= 8;
                this.cy_ = value;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void clearCy() {
                this.bitField0_ &= -9;
                this.cy_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CameraIntrinsicsOrBuilder
            public boolean hasK0() {
                return (this.bitField0_ & 16) != 0;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CameraIntrinsicsOrBuilder
            public float getK0() {
                return this.k0_;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setK0(float value) {
                this.bitField0_ |= 16;
                this.k0_ = value;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void clearK0() {
                this.bitField0_ &= -17;
                this.k0_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CameraIntrinsicsOrBuilder
            public boolean hasK1() {
                return (this.bitField0_ & 32) != 0;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CameraIntrinsicsOrBuilder
            public float getK1() {
                return this.k1_;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setK1(float value) {
                this.bitField0_ |= 32;
                this.k1_ = value;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void clearK1() {
                this.bitField0_ &= -33;
                this.k1_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CameraIntrinsicsOrBuilder
            public boolean hasK2() {
                return (this.bitField0_ & 64) != 0;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CameraIntrinsicsOrBuilder
            public float getK2() {
                return this.k2_;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setK2(float value) {
                this.bitField0_ |= 64;
                this.k2_ = value;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void clearK2() {
                this.bitField0_ &= -65;
                this.k2_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CameraIntrinsicsOrBuilder
            public boolean hasW() {
                return (this.bitField0_ & 128) != 0;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CameraIntrinsicsOrBuilder
            public int getW() {
                return this.w_;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setW(int value) {
                this.bitField0_ |= 128;
                this.w_ = value;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void clearW() {
                this.bitField0_ &= -129;
                this.w_ = 0;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CameraIntrinsicsOrBuilder
            public boolean hasH() {
                return (this.bitField0_ & 256) != 0;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CameraIntrinsicsOrBuilder
            public int getH() {
                return this.h_;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setH(int value) {
                this.bitField0_ |= 256;
                this.h_ = value;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void clearH() {
                this.bitField0_ &= -257;
                this.h_ = 0;
            }

            public static CameraIntrinsics parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
                return (CameraIntrinsics) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
            }

            public static CameraIntrinsics parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
                return (CameraIntrinsics) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
            }

            public static CameraIntrinsics parseFrom(ByteString data) throws InvalidProtocolBufferException {
                return (CameraIntrinsics) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
            }

            public static CameraIntrinsics parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
                return (CameraIntrinsics) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
            }

            public static CameraIntrinsics parseFrom(byte[] data) throws InvalidProtocolBufferException {
                return (CameraIntrinsics) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
            }

            public static CameraIntrinsics parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
                return (CameraIntrinsics) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
            }

            public static CameraIntrinsics parseFrom(InputStream input) throws IOException {
                return (CameraIntrinsics) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
            }

            public static CameraIntrinsics parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
                return (CameraIntrinsics) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
            }

            public static CameraIntrinsics parseDelimitedFrom(InputStream input) throws IOException {
                return (CameraIntrinsics) parseDelimitedFrom(DEFAULT_INSTANCE, input);
            }

            public static CameraIntrinsics parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
                return (CameraIntrinsics) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
            }

            public static CameraIntrinsics parseFrom(CodedInputStream input) throws IOException {
                return (CameraIntrinsics) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
            }

            public static CameraIntrinsics parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
                return (CameraIntrinsics) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
            }

            public static Builder newBuilder() {
                return DEFAULT_INSTANCE.createBuilder();
            }

            public static Builder newBuilder(CameraIntrinsics prototype) {
                return DEFAULT_INSTANCE.createBuilder(prototype);
            }

            /* JADX WARN: Classes with same name are omitted:
              classes2.dex
             */
            /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/TrackingProto$TrackStepOptions$CameraIntrinsics$Builder.class */
            public static final class Builder extends GeneratedMessageLite.Builder<CameraIntrinsics, Builder> implements CameraIntrinsicsOrBuilder {
                private Builder() {
                    super(CameraIntrinsics.DEFAULT_INSTANCE);
                }

                @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CameraIntrinsicsOrBuilder
                public boolean hasFx() {
                    return ((CameraIntrinsics) this.instance).hasFx();
                }

                @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CameraIntrinsicsOrBuilder
                public float getFx() {
                    return ((CameraIntrinsics) this.instance).getFx();
                }

                public Builder setFx(float value) {
                    copyOnWrite();
                    ((CameraIntrinsics) this.instance).setFx(value);
                    return this;
                }

                public Builder clearFx() {
                    copyOnWrite();
                    ((CameraIntrinsics) this.instance).clearFx();
                    return this;
                }

                @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CameraIntrinsicsOrBuilder
                public boolean hasFy() {
                    return ((CameraIntrinsics) this.instance).hasFy();
                }

                @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CameraIntrinsicsOrBuilder
                public float getFy() {
                    return ((CameraIntrinsics) this.instance).getFy();
                }

                public Builder setFy(float value) {
                    copyOnWrite();
                    ((CameraIntrinsics) this.instance).setFy(value);
                    return this;
                }

                public Builder clearFy() {
                    copyOnWrite();
                    ((CameraIntrinsics) this.instance).clearFy();
                    return this;
                }

                @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CameraIntrinsicsOrBuilder
                public boolean hasCx() {
                    return ((CameraIntrinsics) this.instance).hasCx();
                }

                @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CameraIntrinsicsOrBuilder
                public float getCx() {
                    return ((CameraIntrinsics) this.instance).getCx();
                }

                public Builder setCx(float value) {
                    copyOnWrite();
                    ((CameraIntrinsics) this.instance).setCx(value);
                    return this;
                }

                public Builder clearCx() {
                    copyOnWrite();
                    ((CameraIntrinsics) this.instance).clearCx();
                    return this;
                }

                @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CameraIntrinsicsOrBuilder
                public boolean hasCy() {
                    return ((CameraIntrinsics) this.instance).hasCy();
                }

                @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CameraIntrinsicsOrBuilder
                public float getCy() {
                    return ((CameraIntrinsics) this.instance).getCy();
                }

                public Builder setCy(float value) {
                    copyOnWrite();
                    ((CameraIntrinsics) this.instance).setCy(value);
                    return this;
                }

                public Builder clearCy() {
                    copyOnWrite();
                    ((CameraIntrinsics) this.instance).clearCy();
                    return this;
                }

                @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CameraIntrinsicsOrBuilder
                public boolean hasK0() {
                    return ((CameraIntrinsics) this.instance).hasK0();
                }

                @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CameraIntrinsicsOrBuilder
                public float getK0() {
                    return ((CameraIntrinsics) this.instance).getK0();
                }

                public Builder setK0(float value) {
                    copyOnWrite();
                    ((CameraIntrinsics) this.instance).setK0(value);
                    return this;
                }

                public Builder clearK0() {
                    copyOnWrite();
                    ((CameraIntrinsics) this.instance).clearK0();
                    return this;
                }

                @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CameraIntrinsicsOrBuilder
                public boolean hasK1() {
                    return ((CameraIntrinsics) this.instance).hasK1();
                }

                @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CameraIntrinsicsOrBuilder
                public float getK1() {
                    return ((CameraIntrinsics) this.instance).getK1();
                }

                public Builder setK1(float value) {
                    copyOnWrite();
                    ((CameraIntrinsics) this.instance).setK1(value);
                    return this;
                }

                public Builder clearK1() {
                    copyOnWrite();
                    ((CameraIntrinsics) this.instance).clearK1();
                    return this;
                }

                @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CameraIntrinsicsOrBuilder
                public boolean hasK2() {
                    return ((CameraIntrinsics) this.instance).hasK2();
                }

                @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CameraIntrinsicsOrBuilder
                public float getK2() {
                    return ((CameraIntrinsics) this.instance).getK2();
                }

                public Builder setK2(float value) {
                    copyOnWrite();
                    ((CameraIntrinsics) this.instance).setK2(value);
                    return this;
                }

                public Builder clearK2() {
                    copyOnWrite();
                    ((CameraIntrinsics) this.instance).clearK2();
                    return this;
                }

                @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CameraIntrinsicsOrBuilder
                public boolean hasW() {
                    return ((CameraIntrinsics) this.instance).hasW();
                }

                @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CameraIntrinsicsOrBuilder
                public int getW() {
                    return ((CameraIntrinsics) this.instance).getW();
                }

                public Builder setW(int value) {
                    copyOnWrite();
                    ((CameraIntrinsics) this.instance).setW(value);
                    return this;
                }

                public Builder clearW() {
                    copyOnWrite();
                    ((CameraIntrinsics) this.instance).clearW();
                    return this;
                }

                @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CameraIntrinsicsOrBuilder
                public boolean hasH() {
                    return ((CameraIntrinsics) this.instance).hasH();
                }

                @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptions.CameraIntrinsicsOrBuilder
                public int getH() {
                    return ((CameraIntrinsics) this.instance).getH();
                }

                public Builder setH(int value) {
                    copyOnWrite();
                    ((CameraIntrinsics) this.instance).setH(value);
                    return this;
                }

                public Builder clearH() {
                    copyOnWrite();
                    ((CameraIntrinsics) this.instance).clearH();
                    return this;
                }
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.google.protobuf.GeneratedMessageLite
            public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
                switch (method) {
                    case NEW_MUTABLE_INSTANCE:
                        return new CameraIntrinsics();
                    case NEW_BUILDER:
                        return new Builder();
                    case BUILD_MESSAGE_INFO:
                        Object[] objects = {"bitField0_", "fx_", "fy_", "cx_", "cy_", "k0_", "k1_", "k2_", "w_", "h_"};
                        return newMessageInfo(DEFAULT_INSTANCE, "\u0001\t��\u0001\u0001\t\t������\u0001\u0001��\u0002\u0001\u0001\u0003\u0001\u0002\u0004\u0001\u0003\u0005\u0001\u0004\u0006\u0001\u0005\u0007\u0001\u0006\b\u0004\u0007\t\u0004\b", objects);
                    case GET_DEFAULT_INSTANCE:
                        return DEFAULT_INSTANCE;
                    case GET_PARSER:
                        Parser<CameraIntrinsics> parser = PARSER;
                        if (parser == null) {
                            synchronized (CameraIntrinsics.class) {
                                parser = PARSER;
                                if (parser == null) {
                                    parser = new GeneratedMessageLite.DefaultInstanceBasedParser<>(DEFAULT_INSTANCE);
                                    PARSER = parser;
                                }
                            }
                        }
                        return parser;
                    case GET_MEMOIZED_IS_INITIALIZED:
                        return (byte) 1;
                    case SET_MEMOIZED_IS_INITIALIZED:
                        return null;
                    default:
                        throw new UnsupportedOperationException();
                }
            }

            static {
                CameraIntrinsics defaultInstance = new CameraIntrinsics();
                DEFAULT_INSTANCE = defaultInstance;
                GeneratedMessageLite.registerDefaultInstance(CameraIntrinsics.class, defaultInstance);
            }

            public static CameraIntrinsics getDefaultInstance() {
                return DEFAULT_INSTANCE;
            }

            public static Parser<CameraIntrinsics> parser() {
                return DEFAULT_INSTANCE.getParserForType();
            }
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public boolean hasTrackingDegrees() {
            return (this.bitField0_ & 1) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public TrackingDegrees getTrackingDegrees() {
            TrackingDegrees result = TrackingDegrees.forNumber(this.trackingDegrees_);
            return result == null ? TrackingDegrees.TRACKING_DEGREE_TRANSLATION : result;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setTrackingDegrees(TrackingDegrees value) {
            this.trackingDegrees_ = value.getNumber();
            this.bitField0_ |= 1;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearTrackingDegrees() {
            this.bitField0_ &= -2;
            this.trackingDegrees_ = 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public boolean hasTrackObjectAndCamera() {
            return (this.bitField0_ & 2) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public boolean getTrackObjectAndCamera() {
            return this.trackObjectAndCamera_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setTrackObjectAndCamera(boolean value) {
            this.bitField0_ |= 2;
            this.trackObjectAndCamera_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearTrackObjectAndCamera() {
            this.bitField0_ &= -3;
            this.trackObjectAndCamera_ = false;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public boolean hasIrlsIterations() {
            return (this.bitField0_ & 4) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public int getIrlsIterations() {
            return this.irlsIterations_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setIrlsIterations(int value) {
            this.bitField0_ |= 4;
            this.irlsIterations_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearIrlsIterations() {
            this.bitField0_ &= -5;
            this.irlsIterations_ = 5;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public boolean hasSpatialSigma() {
            return (this.bitField0_ & 8) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public float getSpatialSigma() {
            return this.spatialSigma_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setSpatialSigma(float value) {
            this.bitField0_ |= 8;
            this.spatialSigma_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearSpatialSigma() {
            this.bitField0_ &= -9;
            this.spatialSigma_ = 0.15f;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public boolean hasMinMotionSigma() {
            return (this.bitField0_ & 16) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public float getMinMotionSigma() {
            return this.minMotionSigma_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setMinMotionSigma(float value) {
            this.bitField0_ |= 16;
            this.minMotionSigma_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearMinMotionSigma() {
            this.bitField0_ &= -17;
            this.minMotionSigma_ = 0.002f;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public boolean hasRelativeMotionSigma() {
            return (this.bitField0_ & 32) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public float getRelativeMotionSigma() {
            return this.relativeMotionSigma_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setRelativeMotionSigma(float value) {
            this.bitField0_ |= 32;
            this.relativeMotionSigma_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearRelativeMotionSigma() {
            this.bitField0_ &= -33;
            this.relativeMotionSigma_ = 0.3f;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public boolean hasMotionDisparityLowLevel() {
            return (this.bitField0_ & 64) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public float getMotionDisparityLowLevel() {
            return this.motionDisparityLowLevel_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setMotionDisparityLowLevel(float value) {
            this.bitField0_ |= 64;
            this.motionDisparityLowLevel_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearMotionDisparityLowLevel() {
            this.bitField0_ &= -65;
            this.motionDisparityLowLevel_ = 0.008f;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public boolean hasMotionDisparityHighLevel() {
            return (this.bitField0_ & 128) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public float getMotionDisparityHighLevel() {
            return this.motionDisparityHighLevel_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setMotionDisparityHighLevel(float value) {
            this.bitField0_ |= 128;
            this.motionDisparityHighLevel_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearMotionDisparityHighLevel() {
            this.bitField0_ &= -129;
            this.motionDisparityHighLevel_ = 0.016f;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public boolean hasDisparityDecay() {
            return (this.bitField0_ & 256) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public float getDisparityDecay() {
            return this.disparityDecay_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setDisparityDecay(float value) {
            this.bitField0_ |= 256;
            this.disparityDecay_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearDisparityDecay() {
            this.bitField0_ &= -257;
            this.disparityDecay_ = 0.8f;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public boolean hasMotionPriorWeight() {
            return (this.bitField0_ & 512) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public float getMotionPriorWeight() {
            return this.motionPriorWeight_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setMotionPriorWeight(float value) {
            this.bitField0_ |= 512;
            this.motionPriorWeight_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearMotionPriorWeight() {
            this.bitField0_ &= -513;
            this.motionPriorWeight_ = 0.2f;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public boolean hasBackgroundDiscriminationLowLevel() {
            return (this.bitField0_ & 1024) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public float getBackgroundDiscriminationLowLevel() {
            return this.backgroundDiscriminationLowLevel_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setBackgroundDiscriminationLowLevel(float value) {
            this.bitField0_ |= 1024;
            this.backgroundDiscriminationLowLevel_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearBackgroundDiscriminationLowLevel() {
            this.bitField0_ &= -1025;
            this.backgroundDiscriminationLowLevel_ = 0.004f;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public boolean hasBackgroundDiscriminationHighLevel() {
            return (this.bitField0_ & 2048) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public float getBackgroundDiscriminationHighLevel() {
            return this.backgroundDiscriminationHighLevel_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setBackgroundDiscriminationHighLevel(float value) {
            this.bitField0_ |= 2048;
            this.backgroundDiscriminationHighLevel_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearBackgroundDiscriminationHighLevel() {
            this.bitField0_ &= -2049;
            this.backgroundDiscriminationHighLevel_ = 0.008f;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public boolean hasInlierCenterRelativeDistance() {
            return (this.bitField0_ & 4096) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public float getInlierCenterRelativeDistance() {
            return this.inlierCenterRelativeDistance_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setInlierCenterRelativeDistance(float value) {
            this.bitField0_ |= 4096;
            this.inlierCenterRelativeDistance_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearInlierCenterRelativeDistance() {
            this.bitField0_ &= -4097;
            this.inlierCenterRelativeDistance_ = 0.1f;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public boolean hasInlierSpringForce() {
            return (this.bitField0_ & 8192) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public float getInlierSpringForce() {
            return this.inlierSpringForce_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setInlierSpringForce(float value) {
            this.bitField0_ |= 8192;
            this.inlierSpringForce_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearInlierSpringForce() {
            this.bitField0_ &= -8193;
            this.inlierSpringForce_ = 0.3f;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public boolean hasKineticCenterRelativeDistance() {
            return (this.bitField0_ & Calib3d.CALIB_RATIONAL_MODEL) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public float getKineticCenterRelativeDistance() {
            return this.kineticCenterRelativeDistance_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setKineticCenterRelativeDistance(float value) {
            this.bitField0_ |= Calib3d.CALIB_RATIONAL_MODEL;
            this.kineticCenterRelativeDistance_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearKineticCenterRelativeDistance() {
            this.bitField0_ &= -16385;
            this.kineticCenterRelativeDistance_ = 0.4f;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public boolean hasKineticSpringForce() {
            return (this.bitField0_ & Calib3d.CALIB_THIN_PRISM_MODEL) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public float getKineticSpringForce() {
            return this.kineticSpringForce_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setKineticSpringForce(float value) {
            this.bitField0_ |= Calib3d.CALIB_THIN_PRISM_MODEL;
            this.kineticSpringForce_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearKineticSpringForce() {
            this.bitField0_ &= -32769;
            this.kineticSpringForce_ = 0.5f;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public boolean hasKineticSpringForceMinKineticEnergy() {
            return (this.bitField0_ & 65536) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public float getKineticSpringForceMinKineticEnergy() {
            return this.kineticSpringForceMinKineticEnergy_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setKineticSpringForceMinKineticEnergy(float value) {
            this.bitField0_ |= 65536;
            this.kineticSpringForceMinKineticEnergy_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearKineticSpringForceMinKineticEnergy() {
            this.bitField0_ &= -65537;
            this.kineticSpringForceMinKineticEnergy_ = 0.003f;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public boolean hasVelocityUpdateWeight() {
            return (this.bitField0_ & 131072) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public float getVelocityUpdateWeight() {
            return this.velocityUpdateWeight_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setVelocityUpdateWeight(float value) {
            this.bitField0_ |= 131072;
            this.velocityUpdateWeight_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearVelocityUpdateWeight() {
            this.bitField0_ &= -131073;
            this.velocityUpdateWeight_ = 0.7f;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public boolean hasMaxTrackFailures() {
            return (this.bitField0_ & Calib3d.CALIB_TILTED_MODEL) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public int getMaxTrackFailures() {
            return this.maxTrackFailures_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setMaxTrackFailures(int value) {
            this.bitField0_ |= Calib3d.CALIB_TILTED_MODEL;
            this.maxTrackFailures_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearMaxTrackFailures() {
            this.bitField0_ &= -262145;
            this.maxTrackFailures_ = 10;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public boolean hasExpansionSize() {
            return (this.bitField0_ & 524288) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public float getExpansionSize() {
            return this.expansionSize_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setExpansionSize(float value) {
            this.bitField0_ |= 524288;
            this.expansionSize_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearExpansionSize() {
            this.bitField0_ &= -524289;
            this.expansionSize_ = 0.05f;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public boolean hasInlierLowWeight() {
            return (this.bitField0_ & 1048576) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public float getInlierLowWeight() {
            return this.inlierLowWeight_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setInlierLowWeight(float value) {
            this.bitField0_ |= 1048576;
            this.inlierLowWeight_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearInlierLowWeight() {
            this.bitField0_ &= -1048577;
            this.inlierLowWeight_ = 250.0f;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public boolean hasInlierHighWeight() {
            return (this.bitField0_ & Calib3d.CALIB_FIX_TANGENT_DIST) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public float getInlierHighWeight() {
            return this.inlierHighWeight_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setInlierHighWeight(float value) {
            this.bitField0_ |= Calib3d.CALIB_FIX_TANGENT_DIST;
            this.inlierHighWeight_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearInlierHighWeight() {
            this.bitField0_ &= -2097153;
            this.inlierHighWeight_ = 500.0f;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public boolean hasKineticEnergyDecay() {
            return (this.bitField0_ & Calib3d.CALIB_USE_EXTRINSIC_GUESS) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public float getKineticEnergyDecay() {
            return this.kineticEnergyDecay_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setKineticEnergyDecay(float value) {
            this.bitField0_ |= Calib3d.CALIB_USE_EXTRINSIC_GUESS;
            this.kineticEnergyDecay_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearKineticEnergyDecay() {
            this.bitField0_ &= -4194305;
            this.kineticEnergyDecay_ = 0.98f;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public boolean hasPriorWeightIncrease() {
            return (this.bitField0_ & 8388608) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public float getPriorWeightIncrease() {
            return this.priorWeightIncrease_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setPriorWeightIncrease(float value) {
            this.bitField0_ |= 8388608;
            this.priorWeightIncrease_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearPriorWeightIncrease() {
            this.bitField0_ &= -8388609;
            this.priorWeightIncrease_ = 0.2f;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public boolean hasLowKineticEnergy() {
            return (this.bitField0_ & 16777216) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public float getLowKineticEnergy() {
            return this.lowKineticEnergy_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setLowKineticEnergy(float value) {
            this.bitField0_ |= 16777216;
            this.lowKineticEnergy_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearLowKineticEnergy() {
            this.bitField0_ &= -16777217;
            this.lowKineticEnergy_ = 0.001f;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public boolean hasHighKineticEnergy() {
            return (this.bitField0_ & 33554432) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public float getHighKineticEnergy() {
            return this.highKineticEnergy_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setHighKineticEnergy(float value) {
            this.bitField0_ |= 33554432;
            this.highKineticEnergy_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearHighKineticEnergy() {
            this.bitField0_ &= -33554433;
            this.highKineticEnergy_ = 0.004f;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public boolean hasReturnInternalState() {
            return (this.bitField0_ & 67108864) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public boolean getReturnInternalState() {
            return this.returnInternalState_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setReturnInternalState(boolean value) {
            this.bitField0_ |= 67108864;
            this.returnInternalState_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearReturnInternalState() {
            this.bitField0_ &= -67108865;
            this.returnInternalState_ = false;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public boolean hasUsePostEstimationWeightsForState() {
            return (this.bitField0_ & 134217728) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public boolean getUsePostEstimationWeightsForState() {
            return this.usePostEstimationWeightsForState_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setUsePostEstimationWeightsForState(boolean value) {
            this.bitField0_ |= 134217728;
            this.usePostEstimationWeightsForState_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearUsePostEstimationWeightsForState() {
            this.bitField0_ &= -134217729;
            this.usePostEstimationWeightsForState_ = true;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public boolean hasComputeSpatialPrior() {
            return (this.bitField0_ & 268435456) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public boolean getComputeSpatialPrior() {
            return this.computeSpatialPrior_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setComputeSpatialPrior(boolean value) {
            this.bitField0_ |= 268435456;
            this.computeSpatialPrior_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearComputeSpatialPrior() {
            this.bitField0_ &= -268435457;
            this.computeSpatialPrior_ = false;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public boolean hasIrlsInitialization() {
            return (this.bitField0_ & 536870912) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public IrlsInitialization getIrlsInitialization() {
            return this.irlsInitialization_ == null ? IrlsInitialization.getDefaultInstance() : this.irlsInitialization_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setIrlsInitialization(IrlsInitialization value) {
            value.getClass();
            this.irlsInitialization_ = value;
            this.bitField0_ |= 536870912;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void mergeIrlsInitialization(IrlsInitialization value) {
            value.getClass();
            if (this.irlsInitialization_ != null && this.irlsInitialization_ != IrlsInitialization.getDefaultInstance()) {
                this.irlsInitialization_ = IrlsInitialization.newBuilder(this.irlsInitialization_).mergeFrom((IrlsInitialization.Builder) value).buildPartial();
            } else {
                this.irlsInitialization_ = value;
            }
            this.bitField0_ |= 536870912;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearIrlsInitialization() {
            this.irlsInitialization_ = null;
            this.bitField0_ &= -536870913;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public boolean hasStaticMotionTemporalRatio() {
            return (this.bitField0_ & 1073741824) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public float getStaticMotionTemporalRatio() {
            return this.staticMotionTemporalRatio_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setStaticMotionTemporalRatio(float value) {
            this.bitField0_ |= 1073741824;
            this.staticMotionTemporalRatio_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearStaticMotionTemporalRatio() {
            this.bitField0_ &= -1073741825;
            this.staticMotionTemporalRatio_ = 0.003f;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public boolean hasCancelTrackingWithOcclusionOptions() {
            return (this.bitField0_ & Integer.MIN_VALUE) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public CancelTrackingWithOcclusionOptions getCancelTrackingWithOcclusionOptions() {
            return this.cancelTrackingWithOcclusionOptions_ == null ? CancelTrackingWithOcclusionOptions.getDefaultInstance() : this.cancelTrackingWithOcclusionOptions_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setCancelTrackingWithOcclusionOptions(CancelTrackingWithOcclusionOptions value) {
            value.getClass();
            this.cancelTrackingWithOcclusionOptions_ = value;
            this.bitField0_ |= Integer.MIN_VALUE;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void mergeCancelTrackingWithOcclusionOptions(CancelTrackingWithOcclusionOptions value) {
            value.getClass();
            if (this.cancelTrackingWithOcclusionOptions_ != null && this.cancelTrackingWithOcclusionOptions_ != CancelTrackingWithOcclusionOptions.getDefaultInstance()) {
                this.cancelTrackingWithOcclusionOptions_ = CancelTrackingWithOcclusionOptions.newBuilder(this.cancelTrackingWithOcclusionOptions_).mergeFrom((CancelTrackingWithOcclusionOptions.Builder) value).buildPartial();
            } else {
                this.cancelTrackingWithOcclusionOptions_ = value;
            }
            this.bitField0_ |= Integer.MIN_VALUE;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearCancelTrackingWithOcclusionOptions() {
            this.cancelTrackingWithOcclusionOptions_ = null;
            this.bitField0_ &= Integer.MAX_VALUE;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public boolean hasObjectSimilarityMinContdInliers() {
            return (this.bitField1_ & 1) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public int getObjectSimilarityMinContdInliers() {
            return this.objectSimilarityMinContdInliers_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setObjectSimilarityMinContdInliers(int value) {
            this.bitField1_ |= 1;
            this.objectSimilarityMinContdInliers_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearObjectSimilarityMinContdInliers() {
            this.bitField1_ &= -2;
            this.objectSimilarityMinContdInliers_ = 30;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public boolean hasBoxSimilarityMaxScale() {
            return (this.bitField1_ & 2) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public float getBoxSimilarityMaxScale() {
            return this.boxSimilarityMaxScale_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setBoxSimilarityMaxScale(float value) {
            this.bitField1_ |= 2;
            this.boxSimilarityMaxScale_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearBoxSimilarityMaxScale() {
            this.bitField1_ &= -3;
            this.boxSimilarityMaxScale_ = 1.05f;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public boolean hasBoxSimilarityMaxRotation() {
            return (this.bitField1_ & 4) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public float getBoxSimilarityMaxRotation() {
            return this.boxSimilarityMaxRotation_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setBoxSimilarityMaxRotation(float value) {
            this.bitField1_ |= 4;
            this.boxSimilarityMaxRotation_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearBoxSimilarityMaxRotation() {
            this.bitField1_ &= -5;
            this.boxSimilarityMaxRotation_ = 0.2f;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public boolean hasQuadHomographyMaxScale() {
            return (this.bitField1_ & 8) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public float getQuadHomographyMaxScale() {
            return this.quadHomographyMaxScale_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setQuadHomographyMaxScale(float value) {
            this.bitField1_ |= 8;
            this.quadHomographyMaxScale_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearQuadHomographyMaxScale() {
            this.bitField1_ &= -9;
            this.quadHomographyMaxScale_ = 1.2f;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public boolean hasQuadHomographyMaxRotation() {
            return (this.bitField1_ & 16) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public float getQuadHomographyMaxRotation() {
            return this.quadHomographyMaxRotation_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setQuadHomographyMaxRotation(float value) {
            this.bitField1_ |= 16;
            this.quadHomographyMaxRotation_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearQuadHomographyMaxRotation() {
            this.bitField1_ &= -17;
            this.quadHomographyMaxRotation_ = 0.3f;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public boolean hasCameraIntrinsics() {
            return (this.bitField1_ & 32) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public CameraIntrinsics getCameraIntrinsics() {
            return this.cameraIntrinsics_ == null ? CameraIntrinsics.getDefaultInstance() : this.cameraIntrinsics_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setCameraIntrinsics(CameraIntrinsics value) {
            value.getClass();
            this.cameraIntrinsics_ = value;
            this.bitField1_ |= 32;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void mergeCameraIntrinsics(CameraIntrinsics value) {
            value.getClass();
            if (this.cameraIntrinsics_ != null && this.cameraIntrinsics_ != CameraIntrinsics.getDefaultInstance()) {
                this.cameraIntrinsics_ = CameraIntrinsics.newBuilder(this.cameraIntrinsics_).mergeFrom((CameraIntrinsics.Builder) value).buildPartial();
            } else {
                this.cameraIntrinsics_ = value;
            }
            this.bitField1_ |= 32;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearCameraIntrinsics() {
            this.cameraIntrinsics_ = null;
            this.bitField1_ &= -33;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public boolean hasForcedPnpTracking() {
            return (this.bitField1_ & 64) != 0;
        }

        @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
        public boolean getForcedPnpTracking() {
            return this.forcedPnpTracking_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setForcedPnpTracking(boolean value) {
            this.bitField1_ |= 64;
            this.forcedPnpTracking_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearForcedPnpTracking() {
            this.bitField1_ &= -65;
            this.forcedPnpTracking_ = false;
        }

        public static TrackStepOptions parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
            return (TrackStepOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static TrackStepOptions parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (TrackStepOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static TrackStepOptions parseFrom(ByteString data) throws InvalidProtocolBufferException {
            return (TrackStepOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static TrackStepOptions parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (TrackStepOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static TrackStepOptions parseFrom(byte[] data) throws InvalidProtocolBufferException {
            return (TrackStepOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static TrackStepOptions parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (TrackStepOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static TrackStepOptions parseFrom(InputStream input) throws IOException {
            return (TrackStepOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static TrackStepOptions parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (TrackStepOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static TrackStepOptions parseDelimitedFrom(InputStream input) throws IOException {
            return (TrackStepOptions) parseDelimitedFrom(DEFAULT_INSTANCE, input);
        }

        public static TrackStepOptions parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (TrackStepOptions) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static TrackStepOptions parseFrom(CodedInputStream input) throws IOException {
            return (TrackStepOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static TrackStepOptions parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (TrackStepOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Builder newBuilder() {
            return DEFAULT_INSTANCE.createBuilder();
        }

        public static Builder newBuilder(TrackStepOptions prototype) {
            return DEFAULT_INSTANCE.createBuilder(prototype);
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/TrackingProto$TrackStepOptions$Builder.class */
        public static final class Builder extends GeneratedMessageLite.Builder<TrackStepOptions, Builder> implements TrackStepOptionsOrBuilder {
            private Builder() {
                super(TrackStepOptions.DEFAULT_INSTANCE);
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public boolean hasTrackingDegrees() {
                return ((TrackStepOptions) this.instance).hasTrackingDegrees();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public TrackingDegrees getTrackingDegrees() {
                return ((TrackStepOptions) this.instance).getTrackingDegrees();
            }

            public Builder setTrackingDegrees(TrackingDegrees value) {
                copyOnWrite();
                ((TrackStepOptions) this.instance).setTrackingDegrees(value);
                return this;
            }

            public Builder clearTrackingDegrees() {
                copyOnWrite();
                ((TrackStepOptions) this.instance).clearTrackingDegrees();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public boolean hasTrackObjectAndCamera() {
                return ((TrackStepOptions) this.instance).hasTrackObjectAndCamera();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public boolean getTrackObjectAndCamera() {
                return ((TrackStepOptions) this.instance).getTrackObjectAndCamera();
            }

            public Builder setTrackObjectAndCamera(boolean value) {
                copyOnWrite();
                ((TrackStepOptions) this.instance).setTrackObjectAndCamera(value);
                return this;
            }

            public Builder clearTrackObjectAndCamera() {
                copyOnWrite();
                ((TrackStepOptions) this.instance).clearTrackObjectAndCamera();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public boolean hasIrlsIterations() {
                return ((TrackStepOptions) this.instance).hasIrlsIterations();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public int getIrlsIterations() {
                return ((TrackStepOptions) this.instance).getIrlsIterations();
            }

            public Builder setIrlsIterations(int value) {
                copyOnWrite();
                ((TrackStepOptions) this.instance).setIrlsIterations(value);
                return this;
            }

            public Builder clearIrlsIterations() {
                copyOnWrite();
                ((TrackStepOptions) this.instance).clearIrlsIterations();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public boolean hasSpatialSigma() {
                return ((TrackStepOptions) this.instance).hasSpatialSigma();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public float getSpatialSigma() {
                return ((TrackStepOptions) this.instance).getSpatialSigma();
            }

            public Builder setSpatialSigma(float value) {
                copyOnWrite();
                ((TrackStepOptions) this.instance).setSpatialSigma(value);
                return this;
            }

            public Builder clearSpatialSigma() {
                copyOnWrite();
                ((TrackStepOptions) this.instance).clearSpatialSigma();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public boolean hasMinMotionSigma() {
                return ((TrackStepOptions) this.instance).hasMinMotionSigma();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public float getMinMotionSigma() {
                return ((TrackStepOptions) this.instance).getMinMotionSigma();
            }

            public Builder setMinMotionSigma(float value) {
                copyOnWrite();
                ((TrackStepOptions) this.instance).setMinMotionSigma(value);
                return this;
            }

            public Builder clearMinMotionSigma() {
                copyOnWrite();
                ((TrackStepOptions) this.instance).clearMinMotionSigma();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public boolean hasRelativeMotionSigma() {
                return ((TrackStepOptions) this.instance).hasRelativeMotionSigma();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public float getRelativeMotionSigma() {
                return ((TrackStepOptions) this.instance).getRelativeMotionSigma();
            }

            public Builder setRelativeMotionSigma(float value) {
                copyOnWrite();
                ((TrackStepOptions) this.instance).setRelativeMotionSigma(value);
                return this;
            }

            public Builder clearRelativeMotionSigma() {
                copyOnWrite();
                ((TrackStepOptions) this.instance).clearRelativeMotionSigma();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public boolean hasMotionDisparityLowLevel() {
                return ((TrackStepOptions) this.instance).hasMotionDisparityLowLevel();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public float getMotionDisparityLowLevel() {
                return ((TrackStepOptions) this.instance).getMotionDisparityLowLevel();
            }

            public Builder setMotionDisparityLowLevel(float value) {
                copyOnWrite();
                ((TrackStepOptions) this.instance).setMotionDisparityLowLevel(value);
                return this;
            }

            public Builder clearMotionDisparityLowLevel() {
                copyOnWrite();
                ((TrackStepOptions) this.instance).clearMotionDisparityLowLevel();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public boolean hasMotionDisparityHighLevel() {
                return ((TrackStepOptions) this.instance).hasMotionDisparityHighLevel();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public float getMotionDisparityHighLevel() {
                return ((TrackStepOptions) this.instance).getMotionDisparityHighLevel();
            }

            public Builder setMotionDisparityHighLevel(float value) {
                copyOnWrite();
                ((TrackStepOptions) this.instance).setMotionDisparityHighLevel(value);
                return this;
            }

            public Builder clearMotionDisparityHighLevel() {
                copyOnWrite();
                ((TrackStepOptions) this.instance).clearMotionDisparityHighLevel();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public boolean hasDisparityDecay() {
                return ((TrackStepOptions) this.instance).hasDisparityDecay();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public float getDisparityDecay() {
                return ((TrackStepOptions) this.instance).getDisparityDecay();
            }

            public Builder setDisparityDecay(float value) {
                copyOnWrite();
                ((TrackStepOptions) this.instance).setDisparityDecay(value);
                return this;
            }

            public Builder clearDisparityDecay() {
                copyOnWrite();
                ((TrackStepOptions) this.instance).clearDisparityDecay();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public boolean hasMotionPriorWeight() {
                return ((TrackStepOptions) this.instance).hasMotionPriorWeight();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public float getMotionPriorWeight() {
                return ((TrackStepOptions) this.instance).getMotionPriorWeight();
            }

            public Builder setMotionPriorWeight(float value) {
                copyOnWrite();
                ((TrackStepOptions) this.instance).setMotionPriorWeight(value);
                return this;
            }

            public Builder clearMotionPriorWeight() {
                copyOnWrite();
                ((TrackStepOptions) this.instance).clearMotionPriorWeight();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public boolean hasBackgroundDiscriminationLowLevel() {
                return ((TrackStepOptions) this.instance).hasBackgroundDiscriminationLowLevel();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public float getBackgroundDiscriminationLowLevel() {
                return ((TrackStepOptions) this.instance).getBackgroundDiscriminationLowLevel();
            }

            public Builder setBackgroundDiscriminationLowLevel(float value) {
                copyOnWrite();
                ((TrackStepOptions) this.instance).setBackgroundDiscriminationLowLevel(value);
                return this;
            }

            public Builder clearBackgroundDiscriminationLowLevel() {
                copyOnWrite();
                ((TrackStepOptions) this.instance).clearBackgroundDiscriminationLowLevel();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public boolean hasBackgroundDiscriminationHighLevel() {
                return ((TrackStepOptions) this.instance).hasBackgroundDiscriminationHighLevel();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public float getBackgroundDiscriminationHighLevel() {
                return ((TrackStepOptions) this.instance).getBackgroundDiscriminationHighLevel();
            }

            public Builder setBackgroundDiscriminationHighLevel(float value) {
                copyOnWrite();
                ((TrackStepOptions) this.instance).setBackgroundDiscriminationHighLevel(value);
                return this;
            }

            public Builder clearBackgroundDiscriminationHighLevel() {
                copyOnWrite();
                ((TrackStepOptions) this.instance).clearBackgroundDiscriminationHighLevel();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public boolean hasInlierCenterRelativeDistance() {
                return ((TrackStepOptions) this.instance).hasInlierCenterRelativeDistance();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public float getInlierCenterRelativeDistance() {
                return ((TrackStepOptions) this.instance).getInlierCenterRelativeDistance();
            }

            public Builder setInlierCenterRelativeDistance(float value) {
                copyOnWrite();
                ((TrackStepOptions) this.instance).setInlierCenterRelativeDistance(value);
                return this;
            }

            public Builder clearInlierCenterRelativeDistance() {
                copyOnWrite();
                ((TrackStepOptions) this.instance).clearInlierCenterRelativeDistance();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public boolean hasInlierSpringForce() {
                return ((TrackStepOptions) this.instance).hasInlierSpringForce();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public float getInlierSpringForce() {
                return ((TrackStepOptions) this.instance).getInlierSpringForce();
            }

            public Builder setInlierSpringForce(float value) {
                copyOnWrite();
                ((TrackStepOptions) this.instance).setInlierSpringForce(value);
                return this;
            }

            public Builder clearInlierSpringForce() {
                copyOnWrite();
                ((TrackStepOptions) this.instance).clearInlierSpringForce();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public boolean hasKineticCenterRelativeDistance() {
                return ((TrackStepOptions) this.instance).hasKineticCenterRelativeDistance();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public float getKineticCenterRelativeDistance() {
                return ((TrackStepOptions) this.instance).getKineticCenterRelativeDistance();
            }

            public Builder setKineticCenterRelativeDistance(float value) {
                copyOnWrite();
                ((TrackStepOptions) this.instance).setKineticCenterRelativeDistance(value);
                return this;
            }

            public Builder clearKineticCenterRelativeDistance() {
                copyOnWrite();
                ((TrackStepOptions) this.instance).clearKineticCenterRelativeDistance();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public boolean hasKineticSpringForce() {
                return ((TrackStepOptions) this.instance).hasKineticSpringForce();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public float getKineticSpringForce() {
                return ((TrackStepOptions) this.instance).getKineticSpringForce();
            }

            public Builder setKineticSpringForce(float value) {
                copyOnWrite();
                ((TrackStepOptions) this.instance).setKineticSpringForce(value);
                return this;
            }

            public Builder clearKineticSpringForce() {
                copyOnWrite();
                ((TrackStepOptions) this.instance).clearKineticSpringForce();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public boolean hasKineticSpringForceMinKineticEnergy() {
                return ((TrackStepOptions) this.instance).hasKineticSpringForceMinKineticEnergy();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public float getKineticSpringForceMinKineticEnergy() {
                return ((TrackStepOptions) this.instance).getKineticSpringForceMinKineticEnergy();
            }

            public Builder setKineticSpringForceMinKineticEnergy(float value) {
                copyOnWrite();
                ((TrackStepOptions) this.instance).setKineticSpringForceMinKineticEnergy(value);
                return this;
            }

            public Builder clearKineticSpringForceMinKineticEnergy() {
                copyOnWrite();
                ((TrackStepOptions) this.instance).clearKineticSpringForceMinKineticEnergy();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public boolean hasVelocityUpdateWeight() {
                return ((TrackStepOptions) this.instance).hasVelocityUpdateWeight();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public float getVelocityUpdateWeight() {
                return ((TrackStepOptions) this.instance).getVelocityUpdateWeight();
            }

            public Builder setVelocityUpdateWeight(float value) {
                copyOnWrite();
                ((TrackStepOptions) this.instance).setVelocityUpdateWeight(value);
                return this;
            }

            public Builder clearVelocityUpdateWeight() {
                copyOnWrite();
                ((TrackStepOptions) this.instance).clearVelocityUpdateWeight();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public boolean hasMaxTrackFailures() {
                return ((TrackStepOptions) this.instance).hasMaxTrackFailures();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public int getMaxTrackFailures() {
                return ((TrackStepOptions) this.instance).getMaxTrackFailures();
            }

            public Builder setMaxTrackFailures(int value) {
                copyOnWrite();
                ((TrackStepOptions) this.instance).setMaxTrackFailures(value);
                return this;
            }

            public Builder clearMaxTrackFailures() {
                copyOnWrite();
                ((TrackStepOptions) this.instance).clearMaxTrackFailures();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public boolean hasExpansionSize() {
                return ((TrackStepOptions) this.instance).hasExpansionSize();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public float getExpansionSize() {
                return ((TrackStepOptions) this.instance).getExpansionSize();
            }

            public Builder setExpansionSize(float value) {
                copyOnWrite();
                ((TrackStepOptions) this.instance).setExpansionSize(value);
                return this;
            }

            public Builder clearExpansionSize() {
                copyOnWrite();
                ((TrackStepOptions) this.instance).clearExpansionSize();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public boolean hasInlierLowWeight() {
                return ((TrackStepOptions) this.instance).hasInlierLowWeight();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public float getInlierLowWeight() {
                return ((TrackStepOptions) this.instance).getInlierLowWeight();
            }

            public Builder setInlierLowWeight(float value) {
                copyOnWrite();
                ((TrackStepOptions) this.instance).setInlierLowWeight(value);
                return this;
            }

            public Builder clearInlierLowWeight() {
                copyOnWrite();
                ((TrackStepOptions) this.instance).clearInlierLowWeight();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public boolean hasInlierHighWeight() {
                return ((TrackStepOptions) this.instance).hasInlierHighWeight();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public float getInlierHighWeight() {
                return ((TrackStepOptions) this.instance).getInlierHighWeight();
            }

            public Builder setInlierHighWeight(float value) {
                copyOnWrite();
                ((TrackStepOptions) this.instance).setInlierHighWeight(value);
                return this;
            }

            public Builder clearInlierHighWeight() {
                copyOnWrite();
                ((TrackStepOptions) this.instance).clearInlierHighWeight();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public boolean hasKineticEnergyDecay() {
                return ((TrackStepOptions) this.instance).hasKineticEnergyDecay();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public float getKineticEnergyDecay() {
                return ((TrackStepOptions) this.instance).getKineticEnergyDecay();
            }

            public Builder setKineticEnergyDecay(float value) {
                copyOnWrite();
                ((TrackStepOptions) this.instance).setKineticEnergyDecay(value);
                return this;
            }

            public Builder clearKineticEnergyDecay() {
                copyOnWrite();
                ((TrackStepOptions) this.instance).clearKineticEnergyDecay();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public boolean hasPriorWeightIncrease() {
                return ((TrackStepOptions) this.instance).hasPriorWeightIncrease();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public float getPriorWeightIncrease() {
                return ((TrackStepOptions) this.instance).getPriorWeightIncrease();
            }

            public Builder setPriorWeightIncrease(float value) {
                copyOnWrite();
                ((TrackStepOptions) this.instance).setPriorWeightIncrease(value);
                return this;
            }

            public Builder clearPriorWeightIncrease() {
                copyOnWrite();
                ((TrackStepOptions) this.instance).clearPriorWeightIncrease();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public boolean hasLowKineticEnergy() {
                return ((TrackStepOptions) this.instance).hasLowKineticEnergy();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public float getLowKineticEnergy() {
                return ((TrackStepOptions) this.instance).getLowKineticEnergy();
            }

            public Builder setLowKineticEnergy(float value) {
                copyOnWrite();
                ((TrackStepOptions) this.instance).setLowKineticEnergy(value);
                return this;
            }

            public Builder clearLowKineticEnergy() {
                copyOnWrite();
                ((TrackStepOptions) this.instance).clearLowKineticEnergy();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public boolean hasHighKineticEnergy() {
                return ((TrackStepOptions) this.instance).hasHighKineticEnergy();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public float getHighKineticEnergy() {
                return ((TrackStepOptions) this.instance).getHighKineticEnergy();
            }

            public Builder setHighKineticEnergy(float value) {
                copyOnWrite();
                ((TrackStepOptions) this.instance).setHighKineticEnergy(value);
                return this;
            }

            public Builder clearHighKineticEnergy() {
                copyOnWrite();
                ((TrackStepOptions) this.instance).clearHighKineticEnergy();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public boolean hasReturnInternalState() {
                return ((TrackStepOptions) this.instance).hasReturnInternalState();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public boolean getReturnInternalState() {
                return ((TrackStepOptions) this.instance).getReturnInternalState();
            }

            public Builder setReturnInternalState(boolean value) {
                copyOnWrite();
                ((TrackStepOptions) this.instance).setReturnInternalState(value);
                return this;
            }

            public Builder clearReturnInternalState() {
                copyOnWrite();
                ((TrackStepOptions) this.instance).clearReturnInternalState();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public boolean hasUsePostEstimationWeightsForState() {
                return ((TrackStepOptions) this.instance).hasUsePostEstimationWeightsForState();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public boolean getUsePostEstimationWeightsForState() {
                return ((TrackStepOptions) this.instance).getUsePostEstimationWeightsForState();
            }

            public Builder setUsePostEstimationWeightsForState(boolean value) {
                copyOnWrite();
                ((TrackStepOptions) this.instance).setUsePostEstimationWeightsForState(value);
                return this;
            }

            public Builder clearUsePostEstimationWeightsForState() {
                copyOnWrite();
                ((TrackStepOptions) this.instance).clearUsePostEstimationWeightsForState();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public boolean hasComputeSpatialPrior() {
                return ((TrackStepOptions) this.instance).hasComputeSpatialPrior();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public boolean getComputeSpatialPrior() {
                return ((TrackStepOptions) this.instance).getComputeSpatialPrior();
            }

            public Builder setComputeSpatialPrior(boolean value) {
                copyOnWrite();
                ((TrackStepOptions) this.instance).setComputeSpatialPrior(value);
                return this;
            }

            public Builder clearComputeSpatialPrior() {
                copyOnWrite();
                ((TrackStepOptions) this.instance).clearComputeSpatialPrior();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public boolean hasIrlsInitialization() {
                return ((TrackStepOptions) this.instance).hasIrlsInitialization();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public IrlsInitialization getIrlsInitialization() {
                return ((TrackStepOptions) this.instance).getIrlsInitialization();
            }

            public Builder setIrlsInitialization(IrlsInitialization value) {
                copyOnWrite();
                ((TrackStepOptions) this.instance).setIrlsInitialization(value);
                return this;
            }

            public Builder setIrlsInitialization(IrlsInitialization.Builder builderForValue) {
                copyOnWrite();
                ((TrackStepOptions) this.instance).setIrlsInitialization(builderForValue.build());
                return this;
            }

            public Builder mergeIrlsInitialization(IrlsInitialization value) {
                copyOnWrite();
                ((TrackStepOptions) this.instance).mergeIrlsInitialization(value);
                return this;
            }

            public Builder clearIrlsInitialization() {
                copyOnWrite();
                ((TrackStepOptions) this.instance).clearIrlsInitialization();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public boolean hasStaticMotionTemporalRatio() {
                return ((TrackStepOptions) this.instance).hasStaticMotionTemporalRatio();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public float getStaticMotionTemporalRatio() {
                return ((TrackStepOptions) this.instance).getStaticMotionTemporalRatio();
            }

            public Builder setStaticMotionTemporalRatio(float value) {
                copyOnWrite();
                ((TrackStepOptions) this.instance).setStaticMotionTemporalRatio(value);
                return this;
            }

            public Builder clearStaticMotionTemporalRatio() {
                copyOnWrite();
                ((TrackStepOptions) this.instance).clearStaticMotionTemporalRatio();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public boolean hasCancelTrackingWithOcclusionOptions() {
                return ((TrackStepOptions) this.instance).hasCancelTrackingWithOcclusionOptions();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public CancelTrackingWithOcclusionOptions getCancelTrackingWithOcclusionOptions() {
                return ((TrackStepOptions) this.instance).getCancelTrackingWithOcclusionOptions();
            }

            public Builder setCancelTrackingWithOcclusionOptions(CancelTrackingWithOcclusionOptions value) {
                copyOnWrite();
                ((TrackStepOptions) this.instance).setCancelTrackingWithOcclusionOptions(value);
                return this;
            }

            public Builder setCancelTrackingWithOcclusionOptions(CancelTrackingWithOcclusionOptions.Builder builderForValue) {
                copyOnWrite();
                ((TrackStepOptions) this.instance).setCancelTrackingWithOcclusionOptions(builderForValue.build());
                return this;
            }

            public Builder mergeCancelTrackingWithOcclusionOptions(CancelTrackingWithOcclusionOptions value) {
                copyOnWrite();
                ((TrackStepOptions) this.instance).mergeCancelTrackingWithOcclusionOptions(value);
                return this;
            }

            public Builder clearCancelTrackingWithOcclusionOptions() {
                copyOnWrite();
                ((TrackStepOptions) this.instance).clearCancelTrackingWithOcclusionOptions();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public boolean hasObjectSimilarityMinContdInliers() {
                return ((TrackStepOptions) this.instance).hasObjectSimilarityMinContdInliers();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public int getObjectSimilarityMinContdInliers() {
                return ((TrackStepOptions) this.instance).getObjectSimilarityMinContdInliers();
            }

            public Builder setObjectSimilarityMinContdInliers(int value) {
                copyOnWrite();
                ((TrackStepOptions) this.instance).setObjectSimilarityMinContdInliers(value);
                return this;
            }

            public Builder clearObjectSimilarityMinContdInliers() {
                copyOnWrite();
                ((TrackStepOptions) this.instance).clearObjectSimilarityMinContdInliers();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public boolean hasBoxSimilarityMaxScale() {
                return ((TrackStepOptions) this.instance).hasBoxSimilarityMaxScale();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public float getBoxSimilarityMaxScale() {
                return ((TrackStepOptions) this.instance).getBoxSimilarityMaxScale();
            }

            public Builder setBoxSimilarityMaxScale(float value) {
                copyOnWrite();
                ((TrackStepOptions) this.instance).setBoxSimilarityMaxScale(value);
                return this;
            }

            public Builder clearBoxSimilarityMaxScale() {
                copyOnWrite();
                ((TrackStepOptions) this.instance).clearBoxSimilarityMaxScale();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public boolean hasBoxSimilarityMaxRotation() {
                return ((TrackStepOptions) this.instance).hasBoxSimilarityMaxRotation();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public float getBoxSimilarityMaxRotation() {
                return ((TrackStepOptions) this.instance).getBoxSimilarityMaxRotation();
            }

            public Builder setBoxSimilarityMaxRotation(float value) {
                copyOnWrite();
                ((TrackStepOptions) this.instance).setBoxSimilarityMaxRotation(value);
                return this;
            }

            public Builder clearBoxSimilarityMaxRotation() {
                copyOnWrite();
                ((TrackStepOptions) this.instance).clearBoxSimilarityMaxRotation();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public boolean hasQuadHomographyMaxScale() {
                return ((TrackStepOptions) this.instance).hasQuadHomographyMaxScale();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public float getQuadHomographyMaxScale() {
                return ((TrackStepOptions) this.instance).getQuadHomographyMaxScale();
            }

            public Builder setQuadHomographyMaxScale(float value) {
                copyOnWrite();
                ((TrackStepOptions) this.instance).setQuadHomographyMaxScale(value);
                return this;
            }

            public Builder clearQuadHomographyMaxScale() {
                copyOnWrite();
                ((TrackStepOptions) this.instance).clearQuadHomographyMaxScale();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public boolean hasQuadHomographyMaxRotation() {
                return ((TrackStepOptions) this.instance).hasQuadHomographyMaxRotation();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public float getQuadHomographyMaxRotation() {
                return ((TrackStepOptions) this.instance).getQuadHomographyMaxRotation();
            }

            public Builder setQuadHomographyMaxRotation(float value) {
                copyOnWrite();
                ((TrackStepOptions) this.instance).setQuadHomographyMaxRotation(value);
                return this;
            }

            public Builder clearQuadHomographyMaxRotation() {
                copyOnWrite();
                ((TrackStepOptions) this.instance).clearQuadHomographyMaxRotation();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public boolean hasCameraIntrinsics() {
                return ((TrackStepOptions) this.instance).hasCameraIntrinsics();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public CameraIntrinsics getCameraIntrinsics() {
                return ((TrackStepOptions) this.instance).getCameraIntrinsics();
            }

            public Builder setCameraIntrinsics(CameraIntrinsics value) {
                copyOnWrite();
                ((TrackStepOptions) this.instance).setCameraIntrinsics(value);
                return this;
            }

            public Builder setCameraIntrinsics(CameraIntrinsics.Builder builderForValue) {
                copyOnWrite();
                ((TrackStepOptions) this.instance).setCameraIntrinsics(builderForValue.build());
                return this;
            }

            public Builder mergeCameraIntrinsics(CameraIntrinsics value) {
                copyOnWrite();
                ((TrackStepOptions) this.instance).mergeCameraIntrinsics(value);
                return this;
            }

            public Builder clearCameraIntrinsics() {
                copyOnWrite();
                ((TrackStepOptions) this.instance).clearCameraIntrinsics();
                return this;
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public boolean hasForcedPnpTracking() {
                return ((TrackStepOptions) this.instance).hasForcedPnpTracking();
            }

            @Override // com.google.mediapipe.tracking.TrackingProto.TrackStepOptionsOrBuilder
            public boolean getForcedPnpTracking() {
                return ((TrackStepOptions) this.instance).getForcedPnpTracking();
            }

            public Builder setForcedPnpTracking(boolean value) {
                copyOnWrite();
                ((TrackStepOptions) this.instance).setForcedPnpTracking(value);
                return this;
            }

            public Builder clearForcedPnpTracking() {
                copyOnWrite();
                ((TrackStepOptions) this.instance).clearForcedPnpTracking();
                return this;
            }
        }

        /* JADX INFO: Access modifiers changed from: protected */
        @Override // com.google.protobuf.GeneratedMessageLite
        public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
            switch (method) {
                case NEW_MUTABLE_INSTANCE:
                    return new TrackStepOptions();
                case NEW_BUILDER:
                    return new Builder();
                case BUILD_MESSAGE_INFO:
                    Object[] objects = {"bitField0_", "bitField1_", "irlsIterations_", "spatialSigma_", "minMotionSigma_", "relativeMotionSigma_", "motionDisparityLowLevel_", "motionDisparityHighLevel_", "disparityDecay_", "motionPriorWeight_", "backgroundDiscriminationLowLevel_", "backgroundDiscriminationHighLevel_", "inlierCenterRelativeDistance_", "inlierSpringForce_", "kineticCenterRelativeDistance_", "kineticSpringForce_", "velocityUpdateWeight_", "maxTrackFailures_", "expansionSize_", "inlierLowWeight_", "inlierHighWeight_", "kineticSpringForceMinKineticEnergy_", "kineticEnergyDecay_", "priorWeightIncrease_", "lowKineticEnergy_", "highKineticEnergy_", "returnInternalState_", "computeSpatialPrior_", "trackingDegrees_", TrackingDegrees.internalGetVerifier(), "usePostEstimationWeightsForState_", "irlsInitialization_", "trackObjectAndCamera_", "staticMotionTemporalRatio_", "cancelTrackingWithOcclusionOptions_", "objectSimilarityMinContdInliers_", "boxSimilarityMaxScale_", "boxSimilarityMaxRotation_", "quadHomographyMaxScale_", "quadHomographyMaxRotation_", "cameraIntrinsics_", "forcedPnpTracking_"};
                    return newMessageInfo(DEFAULT_INSTANCE, "\u0001'��\u0002\u0001)'������\u0001\u0004\u0002\u0002\u0001\u0003\u0003\u0001\u0004\u0004\u0001\u0005\u0006\u0001\u0006\u0007\u0001\u0007\b\u0001\b\t\u0001\t\n\u0001\n\u000b\u0001\u000b\f\u0001\f\r\u0001\r\u000e\u0001\u000e\u000f\u0001\u000f\u0010\u0001\u0011\u0011\u0004\u0012\u0012\u0001\u0013\u0013\u0001\u0014\u0014\u0001\u0015\u0015\u0001\u0010\u0016\u0001\u0016\u0017\u0001\u0017\u0018\u0001\u0018\u0019\u0001\u0019\u001a\u0007\u001a\u001b\u0007\u001c\u001c\f��\u001d\u0007\u001b\u001e\t\u001d \u0007\u0001!\u0001\u001e\"\t\u001f#\u0004 $\u0001!%\u0001\"&\u0001#'\u0001$(\t%)\u0007&", objects);
                case GET_DEFAULT_INSTANCE:
                    return DEFAULT_INSTANCE;
                case GET_PARSER:
                    Parser<TrackStepOptions> parser = PARSER;
                    if (parser == null) {
                        synchronized (TrackStepOptions.class) {
                            parser = PARSER;
                            if (parser == null) {
                                parser = new GeneratedMessageLite.DefaultInstanceBasedParser<>(DEFAULT_INSTANCE);
                                PARSER = parser;
                            }
                        }
                    }
                    return parser;
                case GET_MEMOIZED_IS_INITIALIZED:
                    return (byte) 1;
                case SET_MEMOIZED_IS_INITIALIZED:
                    return null;
                default:
                    throw new UnsupportedOperationException();
            }
        }

        static {
            TrackStepOptions defaultInstance = new TrackStepOptions();
            DEFAULT_INSTANCE = defaultInstance;
            GeneratedMessageLite.registerDefaultInstance(TrackStepOptions.class, defaultInstance);
        }

        public static TrackStepOptions getDefaultInstance() {
            return DEFAULT_INSTANCE;
        }

        public static Parser<TrackStepOptions> parser() {
            return DEFAULT_INSTANCE.getParserForType();
        }
    }
}