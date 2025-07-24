package com.google.mediapipe.tracking;

import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.mediapipe.tracking.TrackingProto;
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

/* JADX WARN: Classes with same name are omitted:
  classes2.dex
 */
/* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/BoxTrackerProto.class */
public final class BoxTrackerProto {

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/BoxTrackerProto$BoxTrackerOptionsOrBuilder.class */
    public interface BoxTrackerOptionsOrBuilder extends MessageLiteOrBuilder {
        boolean hasCachingChunkSizeMsec();

        int getCachingChunkSizeMsec();

        boolean hasCacheFileFormat();

        String getCacheFileFormat();

        ByteString getCacheFileFormatBytes();

        boolean hasNumTrackingWorkers();

        int getNumTrackingWorkers();

        boolean hasReadChunkTimeoutMsec();

        int getReadChunkTimeoutMsec();

        boolean hasRecordPathStates();

        boolean getRecordPathStates();

        boolean hasTrackStepOptions();

        TrackingProto.TrackStepOptions getTrackStepOptions();
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/BoxTrackerProto$TimedBoxProtoListOrBuilder.class */
    public interface TimedBoxProtoListOrBuilder extends MessageLiteOrBuilder {
        List<TimedBoxProto> getBoxList();

        TimedBoxProto getBox(int index);

        int getBoxCount();
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/BoxTrackerProto$TimedBoxProtoOrBuilder.class */
    public interface TimedBoxProtoOrBuilder extends MessageLiteOrBuilder {
        boolean hasTop();

        float getTop();

        boolean hasLeft();

        float getLeft();

        boolean hasBottom();

        float getBottom();

        boolean hasRight();

        float getRight();

        boolean hasRotation();

        float getRotation();

        boolean hasQuad();

        TrackingProto.MotionBoxState.Quad getQuad();

        boolean hasTimeMsec();

        long getTimeMsec();

        boolean hasId();

        int getId();

        boolean hasLabel();

        String getLabel();

        ByteString getLabelBytes();

        boolean hasConfidence();

        float getConfidence();

        boolean hasAspectRatio();

        float getAspectRatio();

        boolean hasReacquisition();

        boolean getReacquisition();

        boolean hasRequestGrouping();

        boolean getRequestGrouping();
    }

    private BoxTrackerProto() {
    }

    public static void registerAllExtensions(ExtensionRegistryLite registry) {
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/BoxTrackerProto$BoxTrackerOptions.class */
    public static final class BoxTrackerOptions extends GeneratedMessageLite<BoxTrackerOptions, Builder> implements BoxTrackerOptionsOrBuilder {
        private int bitField0_;
        public static final int CACHING_CHUNK_SIZE_MSEC_FIELD_NUMBER = 1;
        public static final int CACHE_FILE_FORMAT_FIELD_NUMBER = 2;
        public static final int NUM_TRACKING_WORKERS_FIELD_NUMBER = 3;
        public static final int READ_CHUNK_TIMEOUT_MSEC_FIELD_NUMBER = 4;
        public static final int RECORD_PATH_STATES_FIELD_NUMBER = 5;
        private boolean recordPathStates_;
        public static final int TRACK_STEP_OPTIONS_FIELD_NUMBER = 6;
        private TrackingProto.TrackStepOptions trackStepOptions_;
        private static final BoxTrackerOptions DEFAULT_INSTANCE;
        private static volatile Parser<BoxTrackerOptions> PARSER;
        private int cachingChunkSizeMsec_ = 2500;
        private String cacheFileFormat_ = "chunk_%04d";
        private int numTrackingWorkers_ = 8;
        private int readChunkTimeoutMsec_ = 60000;

        private BoxTrackerOptions() {
        }

        @Override // com.google.mediapipe.tracking.BoxTrackerProto.BoxTrackerOptionsOrBuilder
        public boolean hasCachingChunkSizeMsec() {
            return (this.bitField0_ & 1) != 0;
        }

        @Override // com.google.mediapipe.tracking.BoxTrackerProto.BoxTrackerOptionsOrBuilder
        public int getCachingChunkSizeMsec() {
            return this.cachingChunkSizeMsec_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setCachingChunkSizeMsec(int value) {
            this.bitField0_ |= 1;
            this.cachingChunkSizeMsec_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearCachingChunkSizeMsec() {
            this.bitField0_ &= -2;
            this.cachingChunkSizeMsec_ = 2500;
        }

        @Override // com.google.mediapipe.tracking.BoxTrackerProto.BoxTrackerOptionsOrBuilder
        public boolean hasCacheFileFormat() {
            return (this.bitField0_ & 2) != 0;
        }

        @Override // com.google.mediapipe.tracking.BoxTrackerProto.BoxTrackerOptionsOrBuilder
        public String getCacheFileFormat() {
            return this.cacheFileFormat_;
        }

        @Override // com.google.mediapipe.tracking.BoxTrackerProto.BoxTrackerOptionsOrBuilder
        public ByteString getCacheFileFormatBytes() {
            return ByteString.copyFromUtf8(this.cacheFileFormat_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setCacheFileFormat(String value) {
            value.getClass();
            this.bitField0_ |= 2;
            this.cacheFileFormat_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearCacheFileFormat() {
            this.bitField0_ &= -3;
            this.cacheFileFormat_ = getDefaultInstance().getCacheFileFormat();
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setCacheFileFormatBytes(ByteString value) {
            this.cacheFileFormat_ = value.toStringUtf8();
            this.bitField0_ |= 2;
        }

        @Override // com.google.mediapipe.tracking.BoxTrackerProto.BoxTrackerOptionsOrBuilder
        public boolean hasNumTrackingWorkers() {
            return (this.bitField0_ & 4) != 0;
        }

        @Override // com.google.mediapipe.tracking.BoxTrackerProto.BoxTrackerOptionsOrBuilder
        public int getNumTrackingWorkers() {
            return this.numTrackingWorkers_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setNumTrackingWorkers(int value) {
            this.bitField0_ |= 4;
            this.numTrackingWorkers_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearNumTrackingWorkers() {
            this.bitField0_ &= -5;
            this.numTrackingWorkers_ = 8;
        }

        @Override // com.google.mediapipe.tracking.BoxTrackerProto.BoxTrackerOptionsOrBuilder
        public boolean hasReadChunkTimeoutMsec() {
            return (this.bitField0_ & 8) != 0;
        }

        @Override // com.google.mediapipe.tracking.BoxTrackerProto.BoxTrackerOptionsOrBuilder
        public int getReadChunkTimeoutMsec() {
            return this.readChunkTimeoutMsec_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setReadChunkTimeoutMsec(int value) {
            this.bitField0_ |= 8;
            this.readChunkTimeoutMsec_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearReadChunkTimeoutMsec() {
            this.bitField0_ &= -9;
            this.readChunkTimeoutMsec_ = 60000;
        }

        @Override // com.google.mediapipe.tracking.BoxTrackerProto.BoxTrackerOptionsOrBuilder
        public boolean hasRecordPathStates() {
            return (this.bitField0_ & 16) != 0;
        }

        @Override // com.google.mediapipe.tracking.BoxTrackerProto.BoxTrackerOptionsOrBuilder
        public boolean getRecordPathStates() {
            return this.recordPathStates_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setRecordPathStates(boolean value) {
            this.bitField0_ |= 16;
            this.recordPathStates_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearRecordPathStates() {
            this.bitField0_ &= -17;
            this.recordPathStates_ = false;
        }

        @Override // com.google.mediapipe.tracking.BoxTrackerProto.BoxTrackerOptionsOrBuilder
        public boolean hasTrackStepOptions() {
            return (this.bitField0_ & 32) != 0;
        }

        @Override // com.google.mediapipe.tracking.BoxTrackerProto.BoxTrackerOptionsOrBuilder
        public TrackingProto.TrackStepOptions getTrackStepOptions() {
            return this.trackStepOptions_ == null ? TrackingProto.TrackStepOptions.getDefaultInstance() : this.trackStepOptions_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setTrackStepOptions(TrackingProto.TrackStepOptions value) {
            value.getClass();
            this.trackStepOptions_ = value;
            this.bitField0_ |= 32;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void mergeTrackStepOptions(TrackingProto.TrackStepOptions value) {
            value.getClass();
            if (this.trackStepOptions_ != null && this.trackStepOptions_ != TrackingProto.TrackStepOptions.getDefaultInstance()) {
                this.trackStepOptions_ = TrackingProto.TrackStepOptions.newBuilder(this.trackStepOptions_).mergeFrom((TrackingProto.TrackStepOptions.Builder) value).buildPartial();
            } else {
                this.trackStepOptions_ = value;
            }
            this.bitField0_ |= 32;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearTrackStepOptions() {
            this.trackStepOptions_ = null;
            this.bitField0_ &= -33;
        }

        public static BoxTrackerOptions parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
            return (BoxTrackerOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static BoxTrackerOptions parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (BoxTrackerOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static BoxTrackerOptions parseFrom(ByteString data) throws InvalidProtocolBufferException {
            return (BoxTrackerOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static BoxTrackerOptions parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (BoxTrackerOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static BoxTrackerOptions parseFrom(byte[] data) throws InvalidProtocolBufferException {
            return (BoxTrackerOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static BoxTrackerOptions parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (BoxTrackerOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static BoxTrackerOptions parseFrom(InputStream input) throws IOException {
            return (BoxTrackerOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static BoxTrackerOptions parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (BoxTrackerOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static BoxTrackerOptions parseDelimitedFrom(InputStream input) throws IOException {
            return (BoxTrackerOptions) parseDelimitedFrom(DEFAULT_INSTANCE, input);
        }

        public static BoxTrackerOptions parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (BoxTrackerOptions) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static BoxTrackerOptions parseFrom(CodedInputStream input) throws IOException {
            return (BoxTrackerOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static BoxTrackerOptions parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (BoxTrackerOptions) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Builder newBuilder() {
            return DEFAULT_INSTANCE.createBuilder();
        }

        public static Builder newBuilder(BoxTrackerOptions prototype) {
            return DEFAULT_INSTANCE.createBuilder(prototype);
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/BoxTrackerProto$BoxTrackerOptions$Builder.class */
        public static final class Builder extends GeneratedMessageLite.Builder<BoxTrackerOptions, Builder> implements BoxTrackerOptionsOrBuilder {
            private Builder() {
                super(BoxTrackerOptions.DEFAULT_INSTANCE);
            }

            @Override // com.google.mediapipe.tracking.BoxTrackerProto.BoxTrackerOptionsOrBuilder
            public boolean hasCachingChunkSizeMsec() {
                return ((BoxTrackerOptions) this.instance).hasCachingChunkSizeMsec();
            }

            @Override // com.google.mediapipe.tracking.BoxTrackerProto.BoxTrackerOptionsOrBuilder
            public int getCachingChunkSizeMsec() {
                return ((BoxTrackerOptions) this.instance).getCachingChunkSizeMsec();
            }

            public Builder setCachingChunkSizeMsec(int value) {
                copyOnWrite();
                ((BoxTrackerOptions) this.instance).setCachingChunkSizeMsec(value);
                return this;
            }

            public Builder clearCachingChunkSizeMsec() {
                copyOnWrite();
                ((BoxTrackerOptions) this.instance).clearCachingChunkSizeMsec();
                return this;
            }

            @Override // com.google.mediapipe.tracking.BoxTrackerProto.BoxTrackerOptionsOrBuilder
            public boolean hasCacheFileFormat() {
                return ((BoxTrackerOptions) this.instance).hasCacheFileFormat();
            }

            @Override // com.google.mediapipe.tracking.BoxTrackerProto.BoxTrackerOptionsOrBuilder
            public String getCacheFileFormat() {
                return ((BoxTrackerOptions) this.instance).getCacheFileFormat();
            }

            @Override // com.google.mediapipe.tracking.BoxTrackerProto.BoxTrackerOptionsOrBuilder
            public ByteString getCacheFileFormatBytes() {
                return ((BoxTrackerOptions) this.instance).getCacheFileFormatBytes();
            }

            public Builder setCacheFileFormat(String value) {
                copyOnWrite();
                ((BoxTrackerOptions) this.instance).setCacheFileFormat(value);
                return this;
            }

            public Builder clearCacheFileFormat() {
                copyOnWrite();
                ((BoxTrackerOptions) this.instance).clearCacheFileFormat();
                return this;
            }

            public Builder setCacheFileFormatBytes(ByteString value) {
                copyOnWrite();
                ((BoxTrackerOptions) this.instance).setCacheFileFormatBytes(value);
                return this;
            }

            @Override // com.google.mediapipe.tracking.BoxTrackerProto.BoxTrackerOptionsOrBuilder
            public boolean hasNumTrackingWorkers() {
                return ((BoxTrackerOptions) this.instance).hasNumTrackingWorkers();
            }

            @Override // com.google.mediapipe.tracking.BoxTrackerProto.BoxTrackerOptionsOrBuilder
            public int getNumTrackingWorkers() {
                return ((BoxTrackerOptions) this.instance).getNumTrackingWorkers();
            }

            public Builder setNumTrackingWorkers(int value) {
                copyOnWrite();
                ((BoxTrackerOptions) this.instance).setNumTrackingWorkers(value);
                return this;
            }

            public Builder clearNumTrackingWorkers() {
                copyOnWrite();
                ((BoxTrackerOptions) this.instance).clearNumTrackingWorkers();
                return this;
            }

            @Override // com.google.mediapipe.tracking.BoxTrackerProto.BoxTrackerOptionsOrBuilder
            public boolean hasReadChunkTimeoutMsec() {
                return ((BoxTrackerOptions) this.instance).hasReadChunkTimeoutMsec();
            }

            @Override // com.google.mediapipe.tracking.BoxTrackerProto.BoxTrackerOptionsOrBuilder
            public int getReadChunkTimeoutMsec() {
                return ((BoxTrackerOptions) this.instance).getReadChunkTimeoutMsec();
            }

            public Builder setReadChunkTimeoutMsec(int value) {
                copyOnWrite();
                ((BoxTrackerOptions) this.instance).setReadChunkTimeoutMsec(value);
                return this;
            }

            public Builder clearReadChunkTimeoutMsec() {
                copyOnWrite();
                ((BoxTrackerOptions) this.instance).clearReadChunkTimeoutMsec();
                return this;
            }

            @Override // com.google.mediapipe.tracking.BoxTrackerProto.BoxTrackerOptionsOrBuilder
            public boolean hasRecordPathStates() {
                return ((BoxTrackerOptions) this.instance).hasRecordPathStates();
            }

            @Override // com.google.mediapipe.tracking.BoxTrackerProto.BoxTrackerOptionsOrBuilder
            public boolean getRecordPathStates() {
                return ((BoxTrackerOptions) this.instance).getRecordPathStates();
            }

            public Builder setRecordPathStates(boolean value) {
                copyOnWrite();
                ((BoxTrackerOptions) this.instance).setRecordPathStates(value);
                return this;
            }

            public Builder clearRecordPathStates() {
                copyOnWrite();
                ((BoxTrackerOptions) this.instance).clearRecordPathStates();
                return this;
            }

            @Override // com.google.mediapipe.tracking.BoxTrackerProto.BoxTrackerOptionsOrBuilder
            public boolean hasTrackStepOptions() {
                return ((BoxTrackerOptions) this.instance).hasTrackStepOptions();
            }

            @Override // com.google.mediapipe.tracking.BoxTrackerProto.BoxTrackerOptionsOrBuilder
            public TrackingProto.TrackStepOptions getTrackStepOptions() {
                return ((BoxTrackerOptions) this.instance).getTrackStepOptions();
            }

            public Builder setTrackStepOptions(TrackingProto.TrackStepOptions value) {
                copyOnWrite();
                ((BoxTrackerOptions) this.instance).setTrackStepOptions(value);
                return this;
            }

            public Builder setTrackStepOptions(TrackingProto.TrackStepOptions.Builder builderForValue) {
                copyOnWrite();
                ((BoxTrackerOptions) this.instance).setTrackStepOptions(builderForValue.build());
                return this;
            }

            public Builder mergeTrackStepOptions(TrackingProto.TrackStepOptions value) {
                copyOnWrite();
                ((BoxTrackerOptions) this.instance).mergeTrackStepOptions(value);
                return this;
            }

            public Builder clearTrackStepOptions() {
                copyOnWrite();
                ((BoxTrackerOptions) this.instance).clearTrackStepOptions();
                return this;
            }
        }

        /* JADX INFO: Access modifiers changed from: protected */
        @Override // com.google.protobuf.GeneratedMessageLite
        public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
            switch (method) {
                case NEW_MUTABLE_INSTANCE:
                    return new BoxTrackerOptions();
                case NEW_BUILDER:
                    return new Builder();
                case BUILD_MESSAGE_INFO:
                    Object[] objects = {"bitField0_", "cachingChunkSizeMsec_", "cacheFileFormat_", "numTrackingWorkers_", "readChunkTimeoutMsec_", "recordPathStates_", "trackStepOptions_"};
                    return newMessageInfo(DEFAULT_INSTANCE, "\u0001\u0006��\u0001\u0001\u0006\u0006������\u0001\u0004��\u0002\b\u0001\u0003\u0004\u0002\u0004\u0004\u0003\u0005\u0007\u0004\u0006\t\u0005", objects);
                case GET_DEFAULT_INSTANCE:
                    return DEFAULT_INSTANCE;
                case GET_PARSER:
                    Parser<BoxTrackerOptions> parser = PARSER;
                    if (parser == null) {
                        synchronized (BoxTrackerOptions.class) {
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
            BoxTrackerOptions defaultInstance = new BoxTrackerOptions();
            DEFAULT_INSTANCE = defaultInstance;
            GeneratedMessageLite.registerDefaultInstance(BoxTrackerOptions.class, defaultInstance);
        }

        public static BoxTrackerOptions getDefaultInstance() {
            return DEFAULT_INSTANCE;
        }

        public static Parser<BoxTrackerOptions> parser() {
            return DEFAULT_INSTANCE.getParserForType();
        }
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/BoxTrackerProto$TimedBoxProto.class */
    public static final class TimedBoxProto extends GeneratedMessageLite<TimedBoxProto, Builder> implements TimedBoxProtoOrBuilder {
        private int bitField0_;
        public static final int TOP_FIELD_NUMBER = 1;
        private float top_;
        public static final int LEFT_FIELD_NUMBER = 2;
        private float left_;
        public static final int BOTTOM_FIELD_NUMBER = 3;
        private float bottom_;
        public static final int RIGHT_FIELD_NUMBER = 4;
        private float right_;
        public static final int ROTATION_FIELD_NUMBER = 7;
        private float rotation_;
        public static final int QUAD_FIELD_NUMBER = 9;
        private TrackingProto.MotionBoxState.Quad quad_;
        public static final int TIME_MSEC_FIELD_NUMBER = 5;
        private long timeMsec_;
        public static final int ID_FIELD_NUMBER = 6;
        public static final int LABEL_FIELD_NUMBER = 13;
        public static final int CONFIDENCE_FIELD_NUMBER = 8;
        private float confidence_;
        public static final int ASPECT_RATIO_FIELD_NUMBER = 10;
        private float aspectRatio_;
        public static final int REACQUISITION_FIELD_NUMBER = 11;
        private boolean reacquisition_;
        public static final int REQUEST_GROUPING_FIELD_NUMBER = 12;
        private boolean requestGrouping_;
        private static final TimedBoxProto DEFAULT_INSTANCE;
        private static volatile Parser<TimedBoxProto> PARSER;
        private int id_ = -1;
        private String label_ = "";

        private TimedBoxProto() {
        }

        @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
        public boolean hasTop() {
            return (this.bitField0_ & 1) != 0;
        }

        @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
        public float getTop() {
            return this.top_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setTop(float value) {
            this.bitField0_ |= 1;
            this.top_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearTop() {
            this.bitField0_ &= -2;
            this.top_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
        public boolean hasLeft() {
            return (this.bitField0_ & 2) != 0;
        }

        @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
        public float getLeft() {
            return this.left_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setLeft(float value) {
            this.bitField0_ |= 2;
            this.left_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearLeft() {
            this.bitField0_ &= -3;
            this.left_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
        public boolean hasBottom() {
            return (this.bitField0_ & 4) != 0;
        }

        @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
        public float getBottom() {
            return this.bottom_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setBottom(float value) {
            this.bitField0_ |= 4;
            this.bottom_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearBottom() {
            this.bitField0_ &= -5;
            this.bottom_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
        public boolean hasRight() {
            return (this.bitField0_ & 8) != 0;
        }

        @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
        public float getRight() {
            return this.right_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setRight(float value) {
            this.bitField0_ |= 8;
            this.right_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearRight() {
            this.bitField0_ &= -9;
            this.right_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
        public boolean hasRotation() {
            return (this.bitField0_ & 16) != 0;
        }

        @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
        public float getRotation() {
            return this.rotation_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setRotation(float value) {
            this.bitField0_ |= 16;
            this.rotation_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearRotation() {
            this.bitField0_ &= -17;
            this.rotation_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
        public boolean hasQuad() {
            return (this.bitField0_ & 32) != 0;
        }

        @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
        public TrackingProto.MotionBoxState.Quad getQuad() {
            return this.quad_ == null ? TrackingProto.MotionBoxState.Quad.getDefaultInstance() : this.quad_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setQuad(TrackingProto.MotionBoxState.Quad value) {
            value.getClass();
            this.quad_ = value;
            this.bitField0_ |= 32;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void mergeQuad(TrackingProto.MotionBoxState.Quad value) {
            value.getClass();
            if (this.quad_ != null && this.quad_ != TrackingProto.MotionBoxState.Quad.getDefaultInstance()) {
                this.quad_ = TrackingProto.MotionBoxState.Quad.newBuilder(this.quad_).mergeFrom((TrackingProto.MotionBoxState.Quad.Builder) value).buildPartial();
            } else {
                this.quad_ = value;
            }
            this.bitField0_ |= 32;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearQuad() {
            this.quad_ = null;
            this.bitField0_ &= -33;
        }

        @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
        public boolean hasTimeMsec() {
            return (this.bitField0_ & 64) != 0;
        }

        @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
        public long getTimeMsec() {
            return this.timeMsec_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setTimeMsec(long value) {
            this.bitField0_ |= 64;
            this.timeMsec_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearTimeMsec() {
            this.bitField0_ &= -65;
            this.timeMsec_ = 0L;
        }

        @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
        public boolean hasId() {
            return (this.bitField0_ & 128) != 0;
        }

        @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
        public int getId() {
            return this.id_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setId(int value) {
            this.bitField0_ |= 128;
            this.id_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearId() {
            this.bitField0_ &= -129;
            this.id_ = -1;
        }

        @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
        public boolean hasLabel() {
            return (this.bitField0_ & 256) != 0;
        }

        @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
        public String getLabel() {
            return this.label_;
        }

        @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
        public ByteString getLabelBytes() {
            return ByteString.copyFromUtf8(this.label_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setLabel(String value) {
            value.getClass();
            this.bitField0_ |= 256;
            this.label_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearLabel() {
            this.bitField0_ &= -257;
            this.label_ = getDefaultInstance().getLabel();
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setLabelBytes(ByteString value) {
            this.label_ = value.toStringUtf8();
            this.bitField0_ |= 256;
        }

        @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
        public boolean hasConfidence() {
            return (this.bitField0_ & 512) != 0;
        }

        @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
        public float getConfidence() {
            return this.confidence_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setConfidence(float value) {
            this.bitField0_ |= 512;
            this.confidence_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearConfidence() {
            this.bitField0_ &= -513;
            this.confidence_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
        public boolean hasAspectRatio() {
            return (this.bitField0_ & 1024) != 0;
        }

        @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
        public float getAspectRatio() {
            return this.aspectRatio_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setAspectRatio(float value) {
            this.bitField0_ |= 1024;
            this.aspectRatio_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearAspectRatio() {
            this.bitField0_ &= -1025;
            this.aspectRatio_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
        public boolean hasReacquisition() {
            return (this.bitField0_ & 2048) != 0;
        }

        @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
        public boolean getReacquisition() {
            return this.reacquisition_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setReacquisition(boolean value) {
            this.bitField0_ |= 2048;
            this.reacquisition_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearReacquisition() {
            this.bitField0_ &= -2049;
            this.reacquisition_ = false;
        }

        @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
        public boolean hasRequestGrouping() {
            return (this.bitField0_ & 4096) != 0;
        }

        @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
        public boolean getRequestGrouping() {
            return this.requestGrouping_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setRequestGrouping(boolean value) {
            this.bitField0_ |= 4096;
            this.requestGrouping_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearRequestGrouping() {
            this.bitField0_ &= -4097;
            this.requestGrouping_ = false;
        }

        public static TimedBoxProto parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
            return (TimedBoxProto) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static TimedBoxProto parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (TimedBoxProto) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static TimedBoxProto parseFrom(ByteString data) throws InvalidProtocolBufferException {
            return (TimedBoxProto) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static TimedBoxProto parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (TimedBoxProto) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static TimedBoxProto parseFrom(byte[] data) throws InvalidProtocolBufferException {
            return (TimedBoxProto) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static TimedBoxProto parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (TimedBoxProto) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static TimedBoxProto parseFrom(InputStream input) throws IOException {
            return (TimedBoxProto) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static TimedBoxProto parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (TimedBoxProto) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static TimedBoxProto parseDelimitedFrom(InputStream input) throws IOException {
            return (TimedBoxProto) parseDelimitedFrom(DEFAULT_INSTANCE, input);
        }

        public static TimedBoxProto parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (TimedBoxProto) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static TimedBoxProto parseFrom(CodedInputStream input) throws IOException {
            return (TimedBoxProto) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static TimedBoxProto parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (TimedBoxProto) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Builder newBuilder() {
            return DEFAULT_INSTANCE.createBuilder();
        }

        public static Builder newBuilder(TimedBoxProto prototype) {
            return DEFAULT_INSTANCE.createBuilder(prototype);
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/BoxTrackerProto$TimedBoxProto$Builder.class */
        public static final class Builder extends GeneratedMessageLite.Builder<TimedBoxProto, Builder> implements TimedBoxProtoOrBuilder {
            private Builder() {
                super(TimedBoxProto.DEFAULT_INSTANCE);
            }

            @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
            public boolean hasTop() {
                return ((TimedBoxProto) this.instance).hasTop();
            }

            @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
            public float getTop() {
                return ((TimedBoxProto) this.instance).getTop();
            }

            public Builder setTop(float value) {
                copyOnWrite();
                ((TimedBoxProto) this.instance).setTop(value);
                return this;
            }

            public Builder clearTop() {
                copyOnWrite();
                ((TimedBoxProto) this.instance).clearTop();
                return this;
            }

            @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
            public boolean hasLeft() {
                return ((TimedBoxProto) this.instance).hasLeft();
            }

            @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
            public float getLeft() {
                return ((TimedBoxProto) this.instance).getLeft();
            }

            public Builder setLeft(float value) {
                copyOnWrite();
                ((TimedBoxProto) this.instance).setLeft(value);
                return this;
            }

            public Builder clearLeft() {
                copyOnWrite();
                ((TimedBoxProto) this.instance).clearLeft();
                return this;
            }

            @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
            public boolean hasBottom() {
                return ((TimedBoxProto) this.instance).hasBottom();
            }

            @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
            public float getBottom() {
                return ((TimedBoxProto) this.instance).getBottom();
            }

            public Builder setBottom(float value) {
                copyOnWrite();
                ((TimedBoxProto) this.instance).setBottom(value);
                return this;
            }

            public Builder clearBottom() {
                copyOnWrite();
                ((TimedBoxProto) this.instance).clearBottom();
                return this;
            }

            @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
            public boolean hasRight() {
                return ((TimedBoxProto) this.instance).hasRight();
            }

            @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
            public float getRight() {
                return ((TimedBoxProto) this.instance).getRight();
            }

            public Builder setRight(float value) {
                copyOnWrite();
                ((TimedBoxProto) this.instance).setRight(value);
                return this;
            }

            public Builder clearRight() {
                copyOnWrite();
                ((TimedBoxProto) this.instance).clearRight();
                return this;
            }

            @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
            public boolean hasRotation() {
                return ((TimedBoxProto) this.instance).hasRotation();
            }

            @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
            public float getRotation() {
                return ((TimedBoxProto) this.instance).getRotation();
            }

            public Builder setRotation(float value) {
                copyOnWrite();
                ((TimedBoxProto) this.instance).setRotation(value);
                return this;
            }

            public Builder clearRotation() {
                copyOnWrite();
                ((TimedBoxProto) this.instance).clearRotation();
                return this;
            }

            @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
            public boolean hasQuad() {
                return ((TimedBoxProto) this.instance).hasQuad();
            }

            @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
            public TrackingProto.MotionBoxState.Quad getQuad() {
                return ((TimedBoxProto) this.instance).getQuad();
            }

            public Builder setQuad(TrackingProto.MotionBoxState.Quad value) {
                copyOnWrite();
                ((TimedBoxProto) this.instance).setQuad(value);
                return this;
            }

            public Builder setQuad(TrackingProto.MotionBoxState.Quad.Builder builderForValue) {
                copyOnWrite();
                ((TimedBoxProto) this.instance).setQuad(builderForValue.build());
                return this;
            }

            public Builder mergeQuad(TrackingProto.MotionBoxState.Quad value) {
                copyOnWrite();
                ((TimedBoxProto) this.instance).mergeQuad(value);
                return this;
            }

            public Builder clearQuad() {
                copyOnWrite();
                ((TimedBoxProto) this.instance).clearQuad();
                return this;
            }

            @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
            public boolean hasTimeMsec() {
                return ((TimedBoxProto) this.instance).hasTimeMsec();
            }

            @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
            public long getTimeMsec() {
                return ((TimedBoxProto) this.instance).getTimeMsec();
            }

            public Builder setTimeMsec(long value) {
                copyOnWrite();
                ((TimedBoxProto) this.instance).setTimeMsec(value);
                return this;
            }

            public Builder clearTimeMsec() {
                copyOnWrite();
                ((TimedBoxProto) this.instance).clearTimeMsec();
                return this;
            }

            @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
            public boolean hasId() {
                return ((TimedBoxProto) this.instance).hasId();
            }

            @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
            public int getId() {
                return ((TimedBoxProto) this.instance).getId();
            }

            public Builder setId(int value) {
                copyOnWrite();
                ((TimedBoxProto) this.instance).setId(value);
                return this;
            }

            public Builder clearId() {
                copyOnWrite();
                ((TimedBoxProto) this.instance).clearId();
                return this;
            }

            @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
            public boolean hasLabel() {
                return ((TimedBoxProto) this.instance).hasLabel();
            }

            @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
            public String getLabel() {
                return ((TimedBoxProto) this.instance).getLabel();
            }

            @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
            public ByteString getLabelBytes() {
                return ((TimedBoxProto) this.instance).getLabelBytes();
            }

            public Builder setLabel(String value) {
                copyOnWrite();
                ((TimedBoxProto) this.instance).setLabel(value);
                return this;
            }

            public Builder clearLabel() {
                copyOnWrite();
                ((TimedBoxProto) this.instance).clearLabel();
                return this;
            }

            public Builder setLabelBytes(ByteString value) {
                copyOnWrite();
                ((TimedBoxProto) this.instance).setLabelBytes(value);
                return this;
            }

            @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
            public boolean hasConfidence() {
                return ((TimedBoxProto) this.instance).hasConfidence();
            }

            @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
            public float getConfidence() {
                return ((TimedBoxProto) this.instance).getConfidence();
            }

            public Builder setConfidence(float value) {
                copyOnWrite();
                ((TimedBoxProto) this.instance).setConfidence(value);
                return this;
            }

            public Builder clearConfidence() {
                copyOnWrite();
                ((TimedBoxProto) this.instance).clearConfidence();
                return this;
            }

            @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
            public boolean hasAspectRatio() {
                return ((TimedBoxProto) this.instance).hasAspectRatio();
            }

            @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
            public float getAspectRatio() {
                return ((TimedBoxProto) this.instance).getAspectRatio();
            }

            public Builder setAspectRatio(float value) {
                copyOnWrite();
                ((TimedBoxProto) this.instance).setAspectRatio(value);
                return this;
            }

            public Builder clearAspectRatio() {
                copyOnWrite();
                ((TimedBoxProto) this.instance).clearAspectRatio();
                return this;
            }

            @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
            public boolean hasReacquisition() {
                return ((TimedBoxProto) this.instance).hasReacquisition();
            }

            @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
            public boolean getReacquisition() {
                return ((TimedBoxProto) this.instance).getReacquisition();
            }

            public Builder setReacquisition(boolean value) {
                copyOnWrite();
                ((TimedBoxProto) this.instance).setReacquisition(value);
                return this;
            }

            public Builder clearReacquisition() {
                copyOnWrite();
                ((TimedBoxProto) this.instance).clearReacquisition();
                return this;
            }

            @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
            public boolean hasRequestGrouping() {
                return ((TimedBoxProto) this.instance).hasRequestGrouping();
            }

            @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoOrBuilder
            public boolean getRequestGrouping() {
                return ((TimedBoxProto) this.instance).getRequestGrouping();
            }

            public Builder setRequestGrouping(boolean value) {
                copyOnWrite();
                ((TimedBoxProto) this.instance).setRequestGrouping(value);
                return this;
            }

            public Builder clearRequestGrouping() {
                copyOnWrite();
                ((TimedBoxProto) this.instance).clearRequestGrouping();
                return this;
            }
        }

        /* JADX INFO: Access modifiers changed from: protected */
        @Override // com.google.protobuf.GeneratedMessageLite
        public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
            switch (method) {
                case NEW_MUTABLE_INSTANCE:
                    return new TimedBoxProto();
                case NEW_BUILDER:
                    return new Builder();
                case BUILD_MESSAGE_INFO:
                    Object[] objects = {"bitField0_", "top_", "left_", "bottom_", "right_", "timeMsec_", "id_", "rotation_", "confidence_", "quad_", "aspectRatio_", "reacquisition_", "requestGrouping_", "label_"};
                    return newMessageInfo(DEFAULT_INSTANCE, "\u0001\r��\u0001\u0001\r\r������\u0001\u0001��\u0002\u0001\u0001\u0003\u0001\u0002\u0004\u0001\u0003\u0005\u0002\u0006\u0006\u0004\u0007\u0007\u0001\u0004\b\u0001\t\t\t\u0005\n\u0001\n\u000b\u0007\u000b\f\u0007\f\r\b\b", objects);
                case GET_DEFAULT_INSTANCE:
                    return DEFAULT_INSTANCE;
                case GET_PARSER:
                    Parser<TimedBoxProto> parser = PARSER;
                    if (parser == null) {
                        synchronized (TimedBoxProto.class) {
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
            TimedBoxProto defaultInstance = new TimedBoxProto();
            DEFAULT_INSTANCE = defaultInstance;
            GeneratedMessageLite.registerDefaultInstance(TimedBoxProto.class, defaultInstance);
        }

        public static TimedBoxProto getDefaultInstance() {
            return DEFAULT_INSTANCE;
        }

        public static Parser<TimedBoxProto> parser() {
            return DEFAULT_INSTANCE.getParserForType();
        }
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/BoxTrackerProto$TimedBoxProtoList.class */
    public static final class TimedBoxProtoList extends GeneratedMessageLite<TimedBoxProtoList, Builder> implements TimedBoxProtoListOrBuilder {
        public static final int BOX_FIELD_NUMBER = 1;
        private Internal.ProtobufList<TimedBoxProto> box_ = emptyProtobufList();
        private static final TimedBoxProtoList DEFAULT_INSTANCE;
        private static volatile Parser<TimedBoxProtoList> PARSER;

        private TimedBoxProtoList() {
        }

        @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoListOrBuilder
        public List<TimedBoxProto> getBoxList() {
            return this.box_;
        }

        public List<? extends TimedBoxProtoOrBuilder> getBoxOrBuilderList() {
            return this.box_;
        }

        @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoListOrBuilder
        public int getBoxCount() {
            return this.box_.size();
        }

        @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoListOrBuilder
        public TimedBoxProto getBox(int index) {
            return this.box_.get(index);
        }

        public TimedBoxProtoOrBuilder getBoxOrBuilder(int index) {
            return this.box_.get(index);
        }

        private void ensureBoxIsMutable() {
            if (!this.box_.isModifiable()) {
                this.box_ = GeneratedMessageLite.mutableCopy(this.box_);
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setBox(int index, TimedBoxProto value) {
            value.getClass();
            ensureBoxIsMutable();
            this.box_.set(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addBox(TimedBoxProto value) {
            value.getClass();
            ensureBoxIsMutable();
            this.box_.add(value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addBox(int index, TimedBoxProto value) {
            value.getClass();
            ensureBoxIsMutable();
            this.box_.add(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addAllBox(Iterable<? extends TimedBoxProto> values) {
            ensureBoxIsMutable();
            AbstractMessageLite.addAll((Iterable) values, (List) this.box_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearBox() {
            this.box_ = emptyProtobufList();
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void removeBox(int index) {
            ensureBoxIsMutable();
            this.box_.remove(index);
        }

        public static TimedBoxProtoList parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
            return (TimedBoxProtoList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static TimedBoxProtoList parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (TimedBoxProtoList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static TimedBoxProtoList parseFrom(ByteString data) throws InvalidProtocolBufferException {
            return (TimedBoxProtoList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static TimedBoxProtoList parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (TimedBoxProtoList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static TimedBoxProtoList parseFrom(byte[] data) throws InvalidProtocolBufferException {
            return (TimedBoxProtoList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static TimedBoxProtoList parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (TimedBoxProtoList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static TimedBoxProtoList parseFrom(InputStream input) throws IOException {
            return (TimedBoxProtoList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static TimedBoxProtoList parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (TimedBoxProtoList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static TimedBoxProtoList parseDelimitedFrom(InputStream input) throws IOException {
            return (TimedBoxProtoList) parseDelimitedFrom(DEFAULT_INSTANCE, input);
        }

        public static TimedBoxProtoList parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (TimedBoxProtoList) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static TimedBoxProtoList parseFrom(CodedInputStream input) throws IOException {
            return (TimedBoxProtoList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static TimedBoxProtoList parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (TimedBoxProtoList) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Builder newBuilder() {
            return DEFAULT_INSTANCE.createBuilder();
        }

        public static Builder newBuilder(TimedBoxProtoList prototype) {
            return DEFAULT_INSTANCE.createBuilder(prototype);
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/BoxTrackerProto$TimedBoxProtoList$Builder.class */
        public static final class Builder extends GeneratedMessageLite.Builder<TimedBoxProtoList, Builder> implements TimedBoxProtoListOrBuilder {
            private Builder() {
                super(TimedBoxProtoList.DEFAULT_INSTANCE);
            }

            @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoListOrBuilder
            public List<TimedBoxProto> getBoxList() {
                return Collections.unmodifiableList(((TimedBoxProtoList) this.instance).getBoxList());
            }

            @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoListOrBuilder
            public int getBoxCount() {
                return ((TimedBoxProtoList) this.instance).getBoxCount();
            }

            @Override // com.google.mediapipe.tracking.BoxTrackerProto.TimedBoxProtoListOrBuilder
            public TimedBoxProto getBox(int index) {
                return ((TimedBoxProtoList) this.instance).getBox(index);
            }

            public Builder setBox(int index, TimedBoxProto value) {
                copyOnWrite();
                ((TimedBoxProtoList) this.instance).setBox(index, value);
                return this;
            }

            public Builder setBox(int index, TimedBoxProto.Builder builderForValue) {
                copyOnWrite();
                ((TimedBoxProtoList) this.instance).setBox(index, builderForValue.build());
                return this;
            }

            public Builder addBox(TimedBoxProto value) {
                copyOnWrite();
                ((TimedBoxProtoList) this.instance).addBox(value);
                return this;
            }

            public Builder addBox(int index, TimedBoxProto value) {
                copyOnWrite();
                ((TimedBoxProtoList) this.instance).addBox(index, value);
                return this;
            }

            public Builder addBox(TimedBoxProto.Builder builderForValue) {
                copyOnWrite();
                ((TimedBoxProtoList) this.instance).addBox(builderForValue.build());
                return this;
            }

            public Builder addBox(int index, TimedBoxProto.Builder builderForValue) {
                copyOnWrite();
                ((TimedBoxProtoList) this.instance).addBox(index, builderForValue.build());
                return this;
            }

            public Builder addAllBox(Iterable<? extends TimedBoxProto> values) {
                copyOnWrite();
                ((TimedBoxProtoList) this.instance).addAllBox(values);
                return this;
            }

            public Builder clearBox() {
                copyOnWrite();
                ((TimedBoxProtoList) this.instance).clearBox();
                return this;
            }

            public Builder removeBox(int index) {
                copyOnWrite();
                ((TimedBoxProtoList) this.instance).removeBox(index);
                return this;
            }
        }

        /* JADX INFO: Access modifiers changed from: protected */
        @Override // com.google.protobuf.GeneratedMessageLite
        public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
            switch (method) {
                case NEW_MUTABLE_INSTANCE:
                    return new TimedBoxProtoList();
                case NEW_BUILDER:
                    return new Builder();
                case BUILD_MESSAGE_INFO:
                    Object[] objects = {"box_", TimedBoxProto.class};
                    return newMessageInfo(DEFAULT_INSTANCE, "\u0001\u0001����\u0001\u0001\u0001��\u0001��\u0001\u001b", objects);
                case GET_DEFAULT_INSTANCE:
                    return DEFAULT_INSTANCE;
                case GET_PARSER:
                    Parser<TimedBoxProtoList> parser = PARSER;
                    if (parser == null) {
                        synchronized (TimedBoxProtoList.class) {
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
            TimedBoxProtoList defaultInstance = new TimedBoxProtoList();
            DEFAULT_INSTANCE = defaultInstance;
            GeneratedMessageLite.registerDefaultInstance(TimedBoxProtoList.class, defaultInstance);
        }

        public static TimedBoxProtoList getDefaultInstance() {
            return DEFAULT_INSTANCE;
        }

        public static Parser<TimedBoxProtoList> parser() {
            return DEFAULT_INSTANCE.getParserForType();
        }
    }
}