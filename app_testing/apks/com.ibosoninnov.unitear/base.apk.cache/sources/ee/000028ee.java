package com.google.mediapipe.formats.proto;

import com.google.android.material.internal.StaticLayoutBuilderCompat;
import com.google.mediapipe.formats.annotation.proto.RasterizationProto;
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
/* loaded from: base.apk:classes.jar:com/google/mediapipe/formats/proto/LocationDataProto.class */
public final class LocationDataProto {

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/formats/proto/LocationDataProto$LocationDataOrBuilder.class */
    public interface LocationDataOrBuilder extends MessageLiteOrBuilder {
        boolean hasFormat();

        LocationData.Format getFormat();

        boolean hasBoundingBox();

        LocationData.BoundingBox getBoundingBox();

        boolean hasRelativeBoundingBox();

        LocationData.RelativeBoundingBox getRelativeBoundingBox();

        boolean hasMask();

        LocationData.BinaryMask getMask();

        List<LocationData.RelativeKeypoint> getRelativeKeypointsList();

        LocationData.RelativeKeypoint getRelativeKeypoints(int index);

        int getRelativeKeypointsCount();
    }

    private LocationDataProto() {
    }

    public static void registerAllExtensions(ExtensionRegistryLite registry) {
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/formats/proto/LocationDataProto$LocationData.class */
    public static final class LocationData extends GeneratedMessageLite<LocationData, Builder> implements LocationDataOrBuilder {
        private int bitField0_;
        public static final int FORMAT_FIELD_NUMBER = 1;
        private int format_;
        public static final int BOUNDING_BOX_FIELD_NUMBER = 2;
        private BoundingBox boundingBox_;
        public static final int RELATIVE_BOUNDING_BOX_FIELD_NUMBER = 3;
        private RelativeBoundingBox relativeBoundingBox_;
        public static final int MASK_FIELD_NUMBER = 4;
        private BinaryMask mask_;
        public static final int RELATIVE_KEYPOINTS_FIELD_NUMBER = 5;
        private static final LocationData DEFAULT_INSTANCE;
        private static volatile Parser<LocationData> PARSER;
        private byte memoizedIsInitialized = 2;
        private Internal.ProtobufList<RelativeKeypoint> relativeKeypoints_ = emptyProtobufList();

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/formats/proto/LocationDataProto$LocationData$BinaryMaskOrBuilder.class */
        public interface BinaryMaskOrBuilder extends MessageLiteOrBuilder {
            boolean hasWidth();

            int getWidth();

            boolean hasHeight();

            int getHeight();

            boolean hasRasterization();

            RasterizationProto.Rasterization getRasterization();
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/formats/proto/LocationDataProto$LocationData$BoundingBoxOrBuilder.class */
        public interface BoundingBoxOrBuilder extends MessageLiteOrBuilder {
            boolean hasXmin();

            int getXmin();

            boolean hasYmin();

            int getYmin();

            boolean hasWidth();

            int getWidth();

            boolean hasHeight();

            int getHeight();
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/formats/proto/LocationDataProto$LocationData$RelativeBoundingBoxOrBuilder.class */
        public interface RelativeBoundingBoxOrBuilder extends MessageLiteOrBuilder {
            boolean hasXmin();

            float getXmin();

            boolean hasYmin();

            float getYmin();

            boolean hasWidth();

            float getWidth();

            boolean hasHeight();

            float getHeight();
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/formats/proto/LocationDataProto$LocationData$RelativeKeypointOrBuilder.class */
        public interface RelativeKeypointOrBuilder extends MessageLiteOrBuilder {
            boolean hasX();

            float getX();

            boolean hasY();

            float getY();

            boolean hasKeypointLabel();

            String getKeypointLabel();

            ByteString getKeypointLabelBytes();

            boolean hasScore();

            float getScore();
        }

        private LocationData() {
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/formats/proto/LocationDataProto$LocationData$Format.class */
        public enum Format implements Internal.EnumLite {
            GLOBAL(0),
            BOUNDING_BOX(1),
            RELATIVE_BOUNDING_BOX(2),
            MASK(3);
            
            public static final int GLOBAL_VALUE = 0;
            public static final int BOUNDING_BOX_VALUE = 1;
            public static final int RELATIVE_BOUNDING_BOX_VALUE = 2;
            public static final int MASK_VALUE = 3;
            private static final Internal.EnumLiteMap<Format> internalValueMap = new Internal.EnumLiteMap<Format>() { // from class: com.google.mediapipe.formats.proto.LocationDataProto.LocationData.Format.1
                /* JADX DEBUG: Method merged with bridge method */
                /* JADX WARN: Can't rename method to resolve collision */
                @Override // com.google.protobuf.Internal.EnumLiteMap
                public Format findValueByNumber(int number) {
                    return Format.forNumber(number);
                }
            };
            private final int value;

            @Override // com.google.protobuf.Internal.EnumLite
            public final int getNumber() {
                return this.value;
            }

            @Deprecated
            public static Format valueOf(int value) {
                return forNumber(value);
            }

            public static Format forNumber(int value) {
                switch (value) {
                    case 0:
                        return GLOBAL;
                    case 1:
                        return BOUNDING_BOX;
                    case 2:
                        return RELATIVE_BOUNDING_BOX;
                    case 3:
                        return MASK;
                    default:
                        return null;
                }
            }

            public static Internal.EnumLiteMap<Format> internalGetValueMap() {
                return internalValueMap;
            }

            public static Internal.EnumVerifier internalGetVerifier() {
                return FormatVerifier.INSTANCE;
            }

            /* JADX INFO: Access modifiers changed from: private */
            /* JADX WARN: Classes with same name are omitted:
              classes2.dex
             */
            /* loaded from: base.apk:classes.jar:com/google/mediapipe/formats/proto/LocationDataProto$LocationData$Format$FormatVerifier.class */
            public static final class FormatVerifier implements Internal.EnumVerifier {
                static final Internal.EnumVerifier INSTANCE = new FormatVerifier();

                private FormatVerifier() {
                }

                @Override // com.google.protobuf.Internal.EnumVerifier
                public boolean isInRange(int number) {
                    return Format.forNumber(number) != null;
                }
            }

            Format(int value) {
                this.value = value;
            }
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/formats/proto/LocationDataProto$LocationData$BoundingBox.class */
        public static final class BoundingBox extends GeneratedMessageLite<BoundingBox, Builder> implements BoundingBoxOrBuilder {
            private int bitField0_;
            public static final int XMIN_FIELD_NUMBER = 1;
            private int xmin_;
            public static final int YMIN_FIELD_NUMBER = 2;
            private int ymin_;
            public static final int WIDTH_FIELD_NUMBER = 3;
            private int width_;
            public static final int HEIGHT_FIELD_NUMBER = 4;
            private int height_;
            private static final BoundingBox DEFAULT_INSTANCE;
            private static volatile Parser<BoundingBox> PARSER;

            private BoundingBox() {
            }

            @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.BoundingBoxOrBuilder
            public boolean hasXmin() {
                return (this.bitField0_ & 1) != 0;
            }

            @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.BoundingBoxOrBuilder
            public int getXmin() {
                return this.xmin_;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setXmin(int value) {
                this.bitField0_ |= 1;
                this.xmin_ = value;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void clearXmin() {
                this.bitField0_ &= -2;
                this.xmin_ = 0;
            }

            @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.BoundingBoxOrBuilder
            public boolean hasYmin() {
                return (this.bitField0_ & 2) != 0;
            }

            @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.BoundingBoxOrBuilder
            public int getYmin() {
                return this.ymin_;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setYmin(int value) {
                this.bitField0_ |= 2;
                this.ymin_ = value;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void clearYmin() {
                this.bitField0_ &= -3;
                this.ymin_ = 0;
            }

            @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.BoundingBoxOrBuilder
            public boolean hasWidth() {
                return (this.bitField0_ & 4) != 0;
            }

            @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.BoundingBoxOrBuilder
            public int getWidth() {
                return this.width_;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setWidth(int value) {
                this.bitField0_ |= 4;
                this.width_ = value;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void clearWidth() {
                this.bitField0_ &= -5;
                this.width_ = 0;
            }

            @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.BoundingBoxOrBuilder
            public boolean hasHeight() {
                return (this.bitField0_ & 8) != 0;
            }

            @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.BoundingBoxOrBuilder
            public int getHeight() {
                return this.height_;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setHeight(int value) {
                this.bitField0_ |= 8;
                this.height_ = value;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void clearHeight() {
                this.bitField0_ &= -9;
                this.height_ = 0;
            }

            public static BoundingBox parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
                return (BoundingBox) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
            }

            public static BoundingBox parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
                return (BoundingBox) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
            }

            public static BoundingBox parseFrom(ByteString data) throws InvalidProtocolBufferException {
                return (BoundingBox) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
            }

            public static BoundingBox parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
                return (BoundingBox) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
            }

            public static BoundingBox parseFrom(byte[] data) throws InvalidProtocolBufferException {
                return (BoundingBox) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
            }

            public static BoundingBox parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
                return (BoundingBox) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
            }

            public static BoundingBox parseFrom(InputStream input) throws IOException {
                return (BoundingBox) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
            }

            public static BoundingBox parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
                return (BoundingBox) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
            }

            public static BoundingBox parseDelimitedFrom(InputStream input) throws IOException {
                return (BoundingBox) parseDelimitedFrom(DEFAULT_INSTANCE, input);
            }

            public static BoundingBox parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
                return (BoundingBox) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
            }

            public static BoundingBox parseFrom(CodedInputStream input) throws IOException {
                return (BoundingBox) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
            }

            public static BoundingBox parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
                return (BoundingBox) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
            }

            public static Builder newBuilder() {
                return DEFAULT_INSTANCE.createBuilder();
            }

            public static Builder newBuilder(BoundingBox prototype) {
                return DEFAULT_INSTANCE.createBuilder(prototype);
            }

            /* JADX WARN: Classes with same name are omitted:
              classes2.dex
             */
            /* loaded from: base.apk:classes.jar:com/google/mediapipe/formats/proto/LocationDataProto$LocationData$BoundingBox$Builder.class */
            public static final class Builder extends GeneratedMessageLite.Builder<BoundingBox, Builder> implements BoundingBoxOrBuilder {
                private Builder() {
                    super(BoundingBox.DEFAULT_INSTANCE);
                }

                @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.BoundingBoxOrBuilder
                public boolean hasXmin() {
                    return ((BoundingBox) this.instance).hasXmin();
                }

                @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.BoundingBoxOrBuilder
                public int getXmin() {
                    return ((BoundingBox) this.instance).getXmin();
                }

                public Builder setXmin(int value) {
                    copyOnWrite();
                    ((BoundingBox) this.instance).setXmin(value);
                    return this;
                }

                public Builder clearXmin() {
                    copyOnWrite();
                    ((BoundingBox) this.instance).clearXmin();
                    return this;
                }

                @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.BoundingBoxOrBuilder
                public boolean hasYmin() {
                    return ((BoundingBox) this.instance).hasYmin();
                }

                @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.BoundingBoxOrBuilder
                public int getYmin() {
                    return ((BoundingBox) this.instance).getYmin();
                }

                public Builder setYmin(int value) {
                    copyOnWrite();
                    ((BoundingBox) this.instance).setYmin(value);
                    return this;
                }

                public Builder clearYmin() {
                    copyOnWrite();
                    ((BoundingBox) this.instance).clearYmin();
                    return this;
                }

                @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.BoundingBoxOrBuilder
                public boolean hasWidth() {
                    return ((BoundingBox) this.instance).hasWidth();
                }

                @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.BoundingBoxOrBuilder
                public int getWidth() {
                    return ((BoundingBox) this.instance).getWidth();
                }

                public Builder setWidth(int value) {
                    copyOnWrite();
                    ((BoundingBox) this.instance).setWidth(value);
                    return this;
                }

                public Builder clearWidth() {
                    copyOnWrite();
                    ((BoundingBox) this.instance).clearWidth();
                    return this;
                }

                @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.BoundingBoxOrBuilder
                public boolean hasHeight() {
                    return ((BoundingBox) this.instance).hasHeight();
                }

                @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.BoundingBoxOrBuilder
                public int getHeight() {
                    return ((BoundingBox) this.instance).getHeight();
                }

                public Builder setHeight(int value) {
                    copyOnWrite();
                    ((BoundingBox) this.instance).setHeight(value);
                    return this;
                }

                public Builder clearHeight() {
                    copyOnWrite();
                    ((BoundingBox) this.instance).clearHeight();
                    return this;
                }
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.google.protobuf.GeneratedMessageLite
            public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
                switch (method) {
                    case NEW_MUTABLE_INSTANCE:
                        return new BoundingBox();
                    case NEW_BUILDER:
                        return new Builder();
                    case BUILD_MESSAGE_INFO:
                        Object[] objects = {"bitField0_", "xmin_", "ymin_", "width_", "height_"};
                        return newMessageInfo(DEFAULT_INSTANCE, "\u0001\u0004��\u0001\u0001\u0004\u0004������\u0001\u0004��\u0002\u0004\u0001\u0003\u0004\u0002\u0004\u0004\u0003", objects);
                    case GET_DEFAULT_INSTANCE:
                        return DEFAULT_INSTANCE;
                    case GET_PARSER:
                        Parser<BoundingBox> parser = PARSER;
                        if (parser == null) {
                            synchronized (BoundingBox.class) {
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
                BoundingBox defaultInstance = new BoundingBox();
                DEFAULT_INSTANCE = defaultInstance;
                GeneratedMessageLite.registerDefaultInstance(BoundingBox.class, defaultInstance);
            }

            public static BoundingBox getDefaultInstance() {
                return DEFAULT_INSTANCE;
            }

            public static Parser<BoundingBox> parser() {
                return DEFAULT_INSTANCE.getParserForType();
            }
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/formats/proto/LocationDataProto$LocationData$RelativeBoundingBox.class */
        public static final class RelativeBoundingBox extends GeneratedMessageLite<RelativeBoundingBox, Builder> implements RelativeBoundingBoxOrBuilder {
            private int bitField0_;
            public static final int XMIN_FIELD_NUMBER = 1;
            private float xmin_;
            public static final int YMIN_FIELD_NUMBER = 2;
            private float ymin_;
            public static final int WIDTH_FIELD_NUMBER = 3;
            private float width_;
            public static final int HEIGHT_FIELD_NUMBER = 4;
            private float height_;
            private static final RelativeBoundingBox DEFAULT_INSTANCE;
            private static volatile Parser<RelativeBoundingBox> PARSER;

            private RelativeBoundingBox() {
            }

            @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.RelativeBoundingBoxOrBuilder
            public boolean hasXmin() {
                return (this.bitField0_ & 1) != 0;
            }

            @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.RelativeBoundingBoxOrBuilder
            public float getXmin() {
                return this.xmin_;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setXmin(float value) {
                this.bitField0_ |= 1;
                this.xmin_ = value;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void clearXmin() {
                this.bitField0_ &= -2;
                this.xmin_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
            }

            @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.RelativeBoundingBoxOrBuilder
            public boolean hasYmin() {
                return (this.bitField0_ & 2) != 0;
            }

            @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.RelativeBoundingBoxOrBuilder
            public float getYmin() {
                return this.ymin_;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setYmin(float value) {
                this.bitField0_ |= 2;
                this.ymin_ = value;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void clearYmin() {
                this.bitField0_ &= -3;
                this.ymin_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
            }

            @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.RelativeBoundingBoxOrBuilder
            public boolean hasWidth() {
                return (this.bitField0_ & 4) != 0;
            }

            @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.RelativeBoundingBoxOrBuilder
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

            @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.RelativeBoundingBoxOrBuilder
            public boolean hasHeight() {
                return (this.bitField0_ & 8) != 0;
            }

            @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.RelativeBoundingBoxOrBuilder
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

            public static RelativeBoundingBox parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
                return (RelativeBoundingBox) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
            }

            public static RelativeBoundingBox parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
                return (RelativeBoundingBox) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
            }

            public static RelativeBoundingBox parseFrom(ByteString data) throws InvalidProtocolBufferException {
                return (RelativeBoundingBox) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
            }

            public static RelativeBoundingBox parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
                return (RelativeBoundingBox) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
            }

            public static RelativeBoundingBox parseFrom(byte[] data) throws InvalidProtocolBufferException {
                return (RelativeBoundingBox) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
            }

            public static RelativeBoundingBox parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
                return (RelativeBoundingBox) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
            }

            public static RelativeBoundingBox parseFrom(InputStream input) throws IOException {
                return (RelativeBoundingBox) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
            }

            public static RelativeBoundingBox parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
                return (RelativeBoundingBox) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
            }

            public static RelativeBoundingBox parseDelimitedFrom(InputStream input) throws IOException {
                return (RelativeBoundingBox) parseDelimitedFrom(DEFAULT_INSTANCE, input);
            }

            public static RelativeBoundingBox parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
                return (RelativeBoundingBox) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
            }

            public static RelativeBoundingBox parseFrom(CodedInputStream input) throws IOException {
                return (RelativeBoundingBox) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
            }

            public static RelativeBoundingBox parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
                return (RelativeBoundingBox) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
            }

            public static Builder newBuilder() {
                return DEFAULT_INSTANCE.createBuilder();
            }

            public static Builder newBuilder(RelativeBoundingBox prototype) {
                return DEFAULT_INSTANCE.createBuilder(prototype);
            }

            /* JADX WARN: Classes with same name are omitted:
              classes2.dex
             */
            /* loaded from: base.apk:classes.jar:com/google/mediapipe/formats/proto/LocationDataProto$LocationData$RelativeBoundingBox$Builder.class */
            public static final class Builder extends GeneratedMessageLite.Builder<RelativeBoundingBox, Builder> implements RelativeBoundingBoxOrBuilder {
                private Builder() {
                    super(RelativeBoundingBox.DEFAULT_INSTANCE);
                }

                @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.RelativeBoundingBoxOrBuilder
                public boolean hasXmin() {
                    return ((RelativeBoundingBox) this.instance).hasXmin();
                }

                @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.RelativeBoundingBoxOrBuilder
                public float getXmin() {
                    return ((RelativeBoundingBox) this.instance).getXmin();
                }

                public Builder setXmin(float value) {
                    copyOnWrite();
                    ((RelativeBoundingBox) this.instance).setXmin(value);
                    return this;
                }

                public Builder clearXmin() {
                    copyOnWrite();
                    ((RelativeBoundingBox) this.instance).clearXmin();
                    return this;
                }

                @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.RelativeBoundingBoxOrBuilder
                public boolean hasYmin() {
                    return ((RelativeBoundingBox) this.instance).hasYmin();
                }

                @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.RelativeBoundingBoxOrBuilder
                public float getYmin() {
                    return ((RelativeBoundingBox) this.instance).getYmin();
                }

                public Builder setYmin(float value) {
                    copyOnWrite();
                    ((RelativeBoundingBox) this.instance).setYmin(value);
                    return this;
                }

                public Builder clearYmin() {
                    copyOnWrite();
                    ((RelativeBoundingBox) this.instance).clearYmin();
                    return this;
                }

                @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.RelativeBoundingBoxOrBuilder
                public boolean hasWidth() {
                    return ((RelativeBoundingBox) this.instance).hasWidth();
                }

                @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.RelativeBoundingBoxOrBuilder
                public float getWidth() {
                    return ((RelativeBoundingBox) this.instance).getWidth();
                }

                public Builder setWidth(float value) {
                    copyOnWrite();
                    ((RelativeBoundingBox) this.instance).setWidth(value);
                    return this;
                }

                public Builder clearWidth() {
                    copyOnWrite();
                    ((RelativeBoundingBox) this.instance).clearWidth();
                    return this;
                }

                @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.RelativeBoundingBoxOrBuilder
                public boolean hasHeight() {
                    return ((RelativeBoundingBox) this.instance).hasHeight();
                }

                @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.RelativeBoundingBoxOrBuilder
                public float getHeight() {
                    return ((RelativeBoundingBox) this.instance).getHeight();
                }

                public Builder setHeight(float value) {
                    copyOnWrite();
                    ((RelativeBoundingBox) this.instance).setHeight(value);
                    return this;
                }

                public Builder clearHeight() {
                    copyOnWrite();
                    ((RelativeBoundingBox) this.instance).clearHeight();
                    return this;
                }
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.google.protobuf.GeneratedMessageLite
            public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
                switch (method) {
                    case NEW_MUTABLE_INSTANCE:
                        return new RelativeBoundingBox();
                    case NEW_BUILDER:
                        return new Builder();
                    case BUILD_MESSAGE_INFO:
                        Object[] objects = {"bitField0_", "xmin_", "ymin_", "width_", "height_"};
                        return newMessageInfo(DEFAULT_INSTANCE, "\u0001\u0004��\u0001\u0001\u0004\u0004������\u0001\u0001��\u0002\u0001\u0001\u0003\u0001\u0002\u0004\u0001\u0003", objects);
                    case GET_DEFAULT_INSTANCE:
                        return DEFAULT_INSTANCE;
                    case GET_PARSER:
                        Parser<RelativeBoundingBox> parser = PARSER;
                        if (parser == null) {
                            synchronized (RelativeBoundingBox.class) {
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
                RelativeBoundingBox defaultInstance = new RelativeBoundingBox();
                DEFAULT_INSTANCE = defaultInstance;
                GeneratedMessageLite.registerDefaultInstance(RelativeBoundingBox.class, defaultInstance);
            }

            public static RelativeBoundingBox getDefaultInstance() {
                return DEFAULT_INSTANCE;
            }

            public static Parser<RelativeBoundingBox> parser() {
                return DEFAULT_INSTANCE.getParserForType();
            }
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/formats/proto/LocationDataProto$LocationData$BinaryMask.class */
        public static final class BinaryMask extends GeneratedMessageLite<BinaryMask, Builder> implements BinaryMaskOrBuilder {
            private int bitField0_;
            public static final int WIDTH_FIELD_NUMBER = 1;
            private int width_;
            public static final int HEIGHT_FIELD_NUMBER = 2;
            private int height_;
            public static final int RASTERIZATION_FIELD_NUMBER = 3;
            private RasterizationProto.Rasterization rasterization_;
            private byte memoizedIsInitialized = 2;
            private static final BinaryMask DEFAULT_INSTANCE;
            private static volatile Parser<BinaryMask> PARSER;

            private BinaryMask() {
            }

            @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.BinaryMaskOrBuilder
            public boolean hasWidth() {
                return (this.bitField0_ & 1) != 0;
            }

            @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.BinaryMaskOrBuilder
            public int getWidth() {
                return this.width_;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setWidth(int value) {
                this.bitField0_ |= 1;
                this.width_ = value;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void clearWidth() {
                this.bitField0_ &= -2;
                this.width_ = 0;
            }

            @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.BinaryMaskOrBuilder
            public boolean hasHeight() {
                return (this.bitField0_ & 2) != 0;
            }

            @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.BinaryMaskOrBuilder
            public int getHeight() {
                return this.height_;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setHeight(int value) {
                this.bitField0_ |= 2;
                this.height_ = value;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void clearHeight() {
                this.bitField0_ &= -3;
                this.height_ = 0;
            }

            @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.BinaryMaskOrBuilder
            public boolean hasRasterization() {
                return (this.bitField0_ & 4) != 0;
            }

            @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.BinaryMaskOrBuilder
            public RasterizationProto.Rasterization getRasterization() {
                return this.rasterization_ == null ? RasterizationProto.Rasterization.getDefaultInstance() : this.rasterization_;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setRasterization(RasterizationProto.Rasterization value) {
                value.getClass();
                this.rasterization_ = value;
                this.bitField0_ |= 4;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void mergeRasterization(RasterizationProto.Rasterization value) {
                value.getClass();
                if (this.rasterization_ != null && this.rasterization_ != RasterizationProto.Rasterization.getDefaultInstance()) {
                    this.rasterization_ = RasterizationProto.Rasterization.newBuilder(this.rasterization_).mergeFrom((RasterizationProto.Rasterization.Builder) value).buildPartial();
                } else {
                    this.rasterization_ = value;
                }
                this.bitField0_ |= 4;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void clearRasterization() {
                this.rasterization_ = null;
                this.bitField0_ &= -5;
            }

            public static BinaryMask parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
                return (BinaryMask) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
            }

            public static BinaryMask parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
                return (BinaryMask) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
            }

            public static BinaryMask parseFrom(ByteString data) throws InvalidProtocolBufferException {
                return (BinaryMask) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
            }

            public static BinaryMask parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
                return (BinaryMask) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
            }

            public static BinaryMask parseFrom(byte[] data) throws InvalidProtocolBufferException {
                return (BinaryMask) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
            }

            public static BinaryMask parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
                return (BinaryMask) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
            }

            public static BinaryMask parseFrom(InputStream input) throws IOException {
                return (BinaryMask) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
            }

            public static BinaryMask parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
                return (BinaryMask) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
            }

            public static BinaryMask parseDelimitedFrom(InputStream input) throws IOException {
                return (BinaryMask) parseDelimitedFrom(DEFAULT_INSTANCE, input);
            }

            public static BinaryMask parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
                return (BinaryMask) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
            }

            public static BinaryMask parseFrom(CodedInputStream input) throws IOException {
                return (BinaryMask) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
            }

            public static BinaryMask parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
                return (BinaryMask) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
            }

            public static Builder newBuilder() {
                return DEFAULT_INSTANCE.createBuilder();
            }

            public static Builder newBuilder(BinaryMask prototype) {
                return DEFAULT_INSTANCE.createBuilder(prototype);
            }

            /* JADX WARN: Classes with same name are omitted:
              classes2.dex
             */
            /* loaded from: base.apk:classes.jar:com/google/mediapipe/formats/proto/LocationDataProto$LocationData$BinaryMask$Builder.class */
            public static final class Builder extends GeneratedMessageLite.Builder<BinaryMask, Builder> implements BinaryMaskOrBuilder {
                private Builder() {
                    super(BinaryMask.DEFAULT_INSTANCE);
                }

                @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.BinaryMaskOrBuilder
                public boolean hasWidth() {
                    return ((BinaryMask) this.instance).hasWidth();
                }

                @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.BinaryMaskOrBuilder
                public int getWidth() {
                    return ((BinaryMask) this.instance).getWidth();
                }

                public Builder setWidth(int value) {
                    copyOnWrite();
                    ((BinaryMask) this.instance).setWidth(value);
                    return this;
                }

                public Builder clearWidth() {
                    copyOnWrite();
                    ((BinaryMask) this.instance).clearWidth();
                    return this;
                }

                @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.BinaryMaskOrBuilder
                public boolean hasHeight() {
                    return ((BinaryMask) this.instance).hasHeight();
                }

                @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.BinaryMaskOrBuilder
                public int getHeight() {
                    return ((BinaryMask) this.instance).getHeight();
                }

                public Builder setHeight(int value) {
                    copyOnWrite();
                    ((BinaryMask) this.instance).setHeight(value);
                    return this;
                }

                public Builder clearHeight() {
                    copyOnWrite();
                    ((BinaryMask) this.instance).clearHeight();
                    return this;
                }

                @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.BinaryMaskOrBuilder
                public boolean hasRasterization() {
                    return ((BinaryMask) this.instance).hasRasterization();
                }

                @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.BinaryMaskOrBuilder
                public RasterizationProto.Rasterization getRasterization() {
                    return ((BinaryMask) this.instance).getRasterization();
                }

                public Builder setRasterization(RasterizationProto.Rasterization value) {
                    copyOnWrite();
                    ((BinaryMask) this.instance).setRasterization(value);
                    return this;
                }

                public Builder setRasterization(RasterizationProto.Rasterization.Builder builderForValue) {
                    copyOnWrite();
                    ((BinaryMask) this.instance).setRasterization(builderForValue.build());
                    return this;
                }

                public Builder mergeRasterization(RasterizationProto.Rasterization value) {
                    copyOnWrite();
                    ((BinaryMask) this.instance).mergeRasterization(value);
                    return this;
                }

                public Builder clearRasterization() {
                    copyOnWrite();
                    ((BinaryMask) this.instance).clearRasterization();
                    return this;
                }
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.google.protobuf.GeneratedMessageLite
            public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
                switch (method) {
                    case NEW_MUTABLE_INSTANCE:
                        return new BinaryMask();
                    case NEW_BUILDER:
                        return new Builder();
                    case BUILD_MESSAGE_INFO:
                        Object[] objects = {"bitField0_", "width_", "height_", "rasterization_"};
                        return newMessageInfo(DEFAULT_INSTANCE, "\u0001\u0003��\u0001\u0001\u0003\u0003����\u0001\u0001\u0004��\u0002\u0004\u0001\u0003Љ\u0002", objects);
                    case GET_DEFAULT_INSTANCE:
                        return DEFAULT_INSTANCE;
                    case GET_PARSER:
                        Parser<BinaryMask> parser = PARSER;
                        if (parser == null) {
                            synchronized (BinaryMask.class) {
                                parser = PARSER;
                                if (parser == null) {
                                    parser = new GeneratedMessageLite.DefaultInstanceBasedParser<>(DEFAULT_INSTANCE);
                                    PARSER = parser;
                                }
                            }
                        }
                        return parser;
                    case GET_MEMOIZED_IS_INITIALIZED:
                        return Byte.valueOf(this.memoizedIsInitialized);
                    case SET_MEMOIZED_IS_INITIALIZED:
                        this.memoizedIsInitialized = (byte) (arg0 == null ? 0 : 1);
                        return null;
                    default:
                        throw new UnsupportedOperationException();
                }
            }

            static {
                BinaryMask defaultInstance = new BinaryMask();
                DEFAULT_INSTANCE = defaultInstance;
                GeneratedMessageLite.registerDefaultInstance(BinaryMask.class, defaultInstance);
            }

            public static BinaryMask getDefaultInstance() {
                return DEFAULT_INSTANCE;
            }

            public static Parser<BinaryMask> parser() {
                return DEFAULT_INSTANCE.getParserForType();
            }
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/formats/proto/LocationDataProto$LocationData$RelativeKeypoint.class */
        public static final class RelativeKeypoint extends GeneratedMessageLite<RelativeKeypoint, Builder> implements RelativeKeypointOrBuilder {
            private int bitField0_;
            public static final int X_FIELD_NUMBER = 1;
            private float x_;
            public static final int Y_FIELD_NUMBER = 2;
            private float y_;
            public static final int KEYPOINT_LABEL_FIELD_NUMBER = 3;
            private String keypointLabel_ = "";
            public static final int SCORE_FIELD_NUMBER = 4;
            private float score_;
            private static final RelativeKeypoint DEFAULT_INSTANCE;
            private static volatile Parser<RelativeKeypoint> PARSER;

            private RelativeKeypoint() {
            }

            @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.RelativeKeypointOrBuilder
            public boolean hasX() {
                return (this.bitField0_ & 1) != 0;
            }

            @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.RelativeKeypointOrBuilder
            public float getX() {
                return this.x_;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setX(float value) {
                this.bitField0_ |= 1;
                this.x_ = value;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void clearX() {
                this.bitField0_ &= -2;
                this.x_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
            }

            @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.RelativeKeypointOrBuilder
            public boolean hasY() {
                return (this.bitField0_ & 2) != 0;
            }

            @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.RelativeKeypointOrBuilder
            public float getY() {
                return this.y_;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setY(float value) {
                this.bitField0_ |= 2;
                this.y_ = value;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void clearY() {
                this.bitField0_ &= -3;
                this.y_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
            }

            @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.RelativeKeypointOrBuilder
            public boolean hasKeypointLabel() {
                return (this.bitField0_ & 4) != 0;
            }

            @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.RelativeKeypointOrBuilder
            public String getKeypointLabel() {
                return this.keypointLabel_;
            }

            @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.RelativeKeypointOrBuilder
            public ByteString getKeypointLabelBytes() {
                return ByteString.copyFromUtf8(this.keypointLabel_);
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setKeypointLabel(String value) {
                value.getClass();
                this.bitField0_ |= 4;
                this.keypointLabel_ = value;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void clearKeypointLabel() {
                this.bitField0_ &= -5;
                this.keypointLabel_ = getDefaultInstance().getKeypointLabel();
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setKeypointLabelBytes(ByteString value) {
                this.keypointLabel_ = value.toStringUtf8();
                this.bitField0_ |= 4;
            }

            @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.RelativeKeypointOrBuilder
            public boolean hasScore() {
                return (this.bitField0_ & 8) != 0;
            }

            @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.RelativeKeypointOrBuilder
            public float getScore() {
                return this.score_;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void setScore(float value) {
                this.bitField0_ |= 8;
                this.score_ = value;
            }

            /* JADX INFO: Access modifiers changed from: private */
            public void clearScore() {
                this.bitField0_ &= -9;
                this.score_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
            }

            public static RelativeKeypoint parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
                return (RelativeKeypoint) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
            }

            public static RelativeKeypoint parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
                return (RelativeKeypoint) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
            }

            public static RelativeKeypoint parseFrom(ByteString data) throws InvalidProtocolBufferException {
                return (RelativeKeypoint) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
            }

            public static RelativeKeypoint parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
                return (RelativeKeypoint) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
            }

            public static RelativeKeypoint parseFrom(byte[] data) throws InvalidProtocolBufferException {
                return (RelativeKeypoint) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
            }

            public static RelativeKeypoint parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
                return (RelativeKeypoint) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
            }

            public static RelativeKeypoint parseFrom(InputStream input) throws IOException {
                return (RelativeKeypoint) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
            }

            public static RelativeKeypoint parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
                return (RelativeKeypoint) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
            }

            public static RelativeKeypoint parseDelimitedFrom(InputStream input) throws IOException {
                return (RelativeKeypoint) parseDelimitedFrom(DEFAULT_INSTANCE, input);
            }

            public static RelativeKeypoint parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
                return (RelativeKeypoint) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
            }

            public static RelativeKeypoint parseFrom(CodedInputStream input) throws IOException {
                return (RelativeKeypoint) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
            }

            public static RelativeKeypoint parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
                return (RelativeKeypoint) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
            }

            public static Builder newBuilder() {
                return DEFAULT_INSTANCE.createBuilder();
            }

            public static Builder newBuilder(RelativeKeypoint prototype) {
                return DEFAULT_INSTANCE.createBuilder(prototype);
            }

            /* JADX WARN: Classes with same name are omitted:
              classes2.dex
             */
            /* loaded from: base.apk:classes.jar:com/google/mediapipe/formats/proto/LocationDataProto$LocationData$RelativeKeypoint$Builder.class */
            public static final class Builder extends GeneratedMessageLite.Builder<RelativeKeypoint, Builder> implements RelativeKeypointOrBuilder {
                private Builder() {
                    super(RelativeKeypoint.DEFAULT_INSTANCE);
                }

                @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.RelativeKeypointOrBuilder
                public boolean hasX() {
                    return ((RelativeKeypoint) this.instance).hasX();
                }

                @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.RelativeKeypointOrBuilder
                public float getX() {
                    return ((RelativeKeypoint) this.instance).getX();
                }

                public Builder setX(float value) {
                    copyOnWrite();
                    ((RelativeKeypoint) this.instance).setX(value);
                    return this;
                }

                public Builder clearX() {
                    copyOnWrite();
                    ((RelativeKeypoint) this.instance).clearX();
                    return this;
                }

                @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.RelativeKeypointOrBuilder
                public boolean hasY() {
                    return ((RelativeKeypoint) this.instance).hasY();
                }

                @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.RelativeKeypointOrBuilder
                public float getY() {
                    return ((RelativeKeypoint) this.instance).getY();
                }

                public Builder setY(float value) {
                    copyOnWrite();
                    ((RelativeKeypoint) this.instance).setY(value);
                    return this;
                }

                public Builder clearY() {
                    copyOnWrite();
                    ((RelativeKeypoint) this.instance).clearY();
                    return this;
                }

                @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.RelativeKeypointOrBuilder
                public boolean hasKeypointLabel() {
                    return ((RelativeKeypoint) this.instance).hasKeypointLabel();
                }

                @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.RelativeKeypointOrBuilder
                public String getKeypointLabel() {
                    return ((RelativeKeypoint) this.instance).getKeypointLabel();
                }

                @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.RelativeKeypointOrBuilder
                public ByteString getKeypointLabelBytes() {
                    return ((RelativeKeypoint) this.instance).getKeypointLabelBytes();
                }

                public Builder setKeypointLabel(String value) {
                    copyOnWrite();
                    ((RelativeKeypoint) this.instance).setKeypointLabel(value);
                    return this;
                }

                public Builder clearKeypointLabel() {
                    copyOnWrite();
                    ((RelativeKeypoint) this.instance).clearKeypointLabel();
                    return this;
                }

                public Builder setKeypointLabelBytes(ByteString value) {
                    copyOnWrite();
                    ((RelativeKeypoint) this.instance).setKeypointLabelBytes(value);
                    return this;
                }

                @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.RelativeKeypointOrBuilder
                public boolean hasScore() {
                    return ((RelativeKeypoint) this.instance).hasScore();
                }

                @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationData.RelativeKeypointOrBuilder
                public float getScore() {
                    return ((RelativeKeypoint) this.instance).getScore();
                }

                public Builder setScore(float value) {
                    copyOnWrite();
                    ((RelativeKeypoint) this.instance).setScore(value);
                    return this;
                }

                public Builder clearScore() {
                    copyOnWrite();
                    ((RelativeKeypoint) this.instance).clearScore();
                    return this;
                }
            }

            /* JADX INFO: Access modifiers changed from: protected */
            @Override // com.google.protobuf.GeneratedMessageLite
            public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
                switch (method) {
                    case NEW_MUTABLE_INSTANCE:
                        return new RelativeKeypoint();
                    case NEW_BUILDER:
                        return new Builder();
                    case BUILD_MESSAGE_INFO:
                        Object[] objects = {"bitField0_", "x_", "y_", "keypointLabel_", "score_"};
                        return newMessageInfo(DEFAULT_INSTANCE, "\u0001\u0004��\u0001\u0001\u0004\u0004������\u0001\u0001��\u0002\u0001\u0001\u0003\b\u0002\u0004\u0001\u0003", objects);
                    case GET_DEFAULT_INSTANCE:
                        return DEFAULT_INSTANCE;
                    case GET_PARSER:
                        Parser<RelativeKeypoint> parser = PARSER;
                        if (parser == null) {
                            synchronized (RelativeKeypoint.class) {
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
                RelativeKeypoint defaultInstance = new RelativeKeypoint();
                DEFAULT_INSTANCE = defaultInstance;
                GeneratedMessageLite.registerDefaultInstance(RelativeKeypoint.class, defaultInstance);
            }

            public static RelativeKeypoint getDefaultInstance() {
                return DEFAULT_INSTANCE;
            }

            public static Parser<RelativeKeypoint> parser() {
                return DEFAULT_INSTANCE.getParserForType();
            }
        }

        @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationDataOrBuilder
        public boolean hasFormat() {
            return (this.bitField0_ & 1) != 0;
        }

        @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationDataOrBuilder
        public Format getFormat() {
            Format result = Format.forNumber(this.format_);
            return result == null ? Format.GLOBAL : result;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setFormat(Format value) {
            this.format_ = value.getNumber();
            this.bitField0_ |= 1;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearFormat() {
            this.bitField0_ &= -2;
            this.format_ = 0;
        }

        @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationDataOrBuilder
        public boolean hasBoundingBox() {
            return (this.bitField0_ & 2) != 0;
        }

        @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationDataOrBuilder
        public BoundingBox getBoundingBox() {
            return this.boundingBox_ == null ? BoundingBox.getDefaultInstance() : this.boundingBox_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setBoundingBox(BoundingBox value) {
            value.getClass();
            this.boundingBox_ = value;
            this.bitField0_ |= 2;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void mergeBoundingBox(BoundingBox value) {
            value.getClass();
            if (this.boundingBox_ != null && this.boundingBox_ != BoundingBox.getDefaultInstance()) {
                this.boundingBox_ = BoundingBox.newBuilder(this.boundingBox_).mergeFrom((BoundingBox.Builder) value).buildPartial();
            } else {
                this.boundingBox_ = value;
            }
            this.bitField0_ |= 2;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearBoundingBox() {
            this.boundingBox_ = null;
            this.bitField0_ &= -3;
        }

        @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationDataOrBuilder
        public boolean hasRelativeBoundingBox() {
            return (this.bitField0_ & 4) != 0;
        }

        @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationDataOrBuilder
        public RelativeBoundingBox getRelativeBoundingBox() {
            return this.relativeBoundingBox_ == null ? RelativeBoundingBox.getDefaultInstance() : this.relativeBoundingBox_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setRelativeBoundingBox(RelativeBoundingBox value) {
            value.getClass();
            this.relativeBoundingBox_ = value;
            this.bitField0_ |= 4;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void mergeRelativeBoundingBox(RelativeBoundingBox value) {
            value.getClass();
            if (this.relativeBoundingBox_ != null && this.relativeBoundingBox_ != RelativeBoundingBox.getDefaultInstance()) {
                this.relativeBoundingBox_ = RelativeBoundingBox.newBuilder(this.relativeBoundingBox_).mergeFrom((RelativeBoundingBox.Builder) value).buildPartial();
            } else {
                this.relativeBoundingBox_ = value;
            }
            this.bitField0_ |= 4;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearRelativeBoundingBox() {
            this.relativeBoundingBox_ = null;
            this.bitField0_ &= -5;
        }

        @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationDataOrBuilder
        public boolean hasMask() {
            return (this.bitField0_ & 8) != 0;
        }

        @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationDataOrBuilder
        public BinaryMask getMask() {
            return this.mask_ == null ? BinaryMask.getDefaultInstance() : this.mask_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setMask(BinaryMask value) {
            value.getClass();
            this.mask_ = value;
            this.bitField0_ |= 8;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void mergeMask(BinaryMask value) {
            value.getClass();
            if (this.mask_ != null && this.mask_ != BinaryMask.getDefaultInstance()) {
                this.mask_ = BinaryMask.newBuilder(this.mask_).mergeFrom((BinaryMask.Builder) value).buildPartial();
            } else {
                this.mask_ = value;
            }
            this.bitField0_ |= 8;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearMask() {
            this.mask_ = null;
            this.bitField0_ &= -9;
        }

        @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationDataOrBuilder
        public List<RelativeKeypoint> getRelativeKeypointsList() {
            return this.relativeKeypoints_;
        }

        public List<? extends RelativeKeypointOrBuilder> getRelativeKeypointsOrBuilderList() {
            return this.relativeKeypoints_;
        }

        @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationDataOrBuilder
        public int getRelativeKeypointsCount() {
            return this.relativeKeypoints_.size();
        }

        @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationDataOrBuilder
        public RelativeKeypoint getRelativeKeypoints(int index) {
            return this.relativeKeypoints_.get(index);
        }

        public RelativeKeypointOrBuilder getRelativeKeypointsOrBuilder(int index) {
            return this.relativeKeypoints_.get(index);
        }

        private void ensureRelativeKeypointsIsMutable() {
            if (!this.relativeKeypoints_.isModifiable()) {
                this.relativeKeypoints_ = GeneratedMessageLite.mutableCopy(this.relativeKeypoints_);
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setRelativeKeypoints(int index, RelativeKeypoint value) {
            value.getClass();
            ensureRelativeKeypointsIsMutable();
            this.relativeKeypoints_.set(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addRelativeKeypoints(RelativeKeypoint value) {
            value.getClass();
            ensureRelativeKeypointsIsMutable();
            this.relativeKeypoints_.add(value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addRelativeKeypoints(int index, RelativeKeypoint value) {
            value.getClass();
            ensureRelativeKeypointsIsMutable();
            this.relativeKeypoints_.add(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addAllRelativeKeypoints(Iterable<? extends RelativeKeypoint> values) {
            ensureRelativeKeypointsIsMutable();
            AbstractMessageLite.addAll((Iterable) values, (List) this.relativeKeypoints_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearRelativeKeypoints() {
            this.relativeKeypoints_ = emptyProtobufList();
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void removeRelativeKeypoints(int index) {
            ensureRelativeKeypointsIsMutable();
            this.relativeKeypoints_.remove(index);
        }

        public static LocationData parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
            return (LocationData) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static LocationData parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (LocationData) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static LocationData parseFrom(ByteString data) throws InvalidProtocolBufferException {
            return (LocationData) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static LocationData parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (LocationData) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static LocationData parseFrom(byte[] data) throws InvalidProtocolBufferException {
            return (LocationData) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static LocationData parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (LocationData) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static LocationData parseFrom(InputStream input) throws IOException {
            return (LocationData) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static LocationData parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (LocationData) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static LocationData parseDelimitedFrom(InputStream input) throws IOException {
            return (LocationData) parseDelimitedFrom(DEFAULT_INSTANCE, input);
        }

        public static LocationData parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (LocationData) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static LocationData parseFrom(CodedInputStream input) throws IOException {
            return (LocationData) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static LocationData parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (LocationData) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Builder newBuilder() {
            return DEFAULT_INSTANCE.createBuilder();
        }

        public static Builder newBuilder(LocationData prototype) {
            return DEFAULT_INSTANCE.createBuilder(prototype);
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/formats/proto/LocationDataProto$LocationData$Builder.class */
        public static final class Builder extends GeneratedMessageLite.Builder<LocationData, Builder> implements LocationDataOrBuilder {
            private Builder() {
                super(LocationData.DEFAULT_INSTANCE);
            }

            @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationDataOrBuilder
            public boolean hasFormat() {
                return ((LocationData) this.instance).hasFormat();
            }

            @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationDataOrBuilder
            public Format getFormat() {
                return ((LocationData) this.instance).getFormat();
            }

            public Builder setFormat(Format value) {
                copyOnWrite();
                ((LocationData) this.instance).setFormat(value);
                return this;
            }

            public Builder clearFormat() {
                copyOnWrite();
                ((LocationData) this.instance).clearFormat();
                return this;
            }

            @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationDataOrBuilder
            public boolean hasBoundingBox() {
                return ((LocationData) this.instance).hasBoundingBox();
            }

            @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationDataOrBuilder
            public BoundingBox getBoundingBox() {
                return ((LocationData) this.instance).getBoundingBox();
            }

            public Builder setBoundingBox(BoundingBox value) {
                copyOnWrite();
                ((LocationData) this.instance).setBoundingBox(value);
                return this;
            }

            public Builder setBoundingBox(BoundingBox.Builder builderForValue) {
                copyOnWrite();
                ((LocationData) this.instance).setBoundingBox(builderForValue.build());
                return this;
            }

            public Builder mergeBoundingBox(BoundingBox value) {
                copyOnWrite();
                ((LocationData) this.instance).mergeBoundingBox(value);
                return this;
            }

            public Builder clearBoundingBox() {
                copyOnWrite();
                ((LocationData) this.instance).clearBoundingBox();
                return this;
            }

            @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationDataOrBuilder
            public boolean hasRelativeBoundingBox() {
                return ((LocationData) this.instance).hasRelativeBoundingBox();
            }

            @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationDataOrBuilder
            public RelativeBoundingBox getRelativeBoundingBox() {
                return ((LocationData) this.instance).getRelativeBoundingBox();
            }

            public Builder setRelativeBoundingBox(RelativeBoundingBox value) {
                copyOnWrite();
                ((LocationData) this.instance).setRelativeBoundingBox(value);
                return this;
            }

            public Builder setRelativeBoundingBox(RelativeBoundingBox.Builder builderForValue) {
                copyOnWrite();
                ((LocationData) this.instance).setRelativeBoundingBox(builderForValue.build());
                return this;
            }

            public Builder mergeRelativeBoundingBox(RelativeBoundingBox value) {
                copyOnWrite();
                ((LocationData) this.instance).mergeRelativeBoundingBox(value);
                return this;
            }

            public Builder clearRelativeBoundingBox() {
                copyOnWrite();
                ((LocationData) this.instance).clearRelativeBoundingBox();
                return this;
            }

            @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationDataOrBuilder
            public boolean hasMask() {
                return ((LocationData) this.instance).hasMask();
            }

            @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationDataOrBuilder
            public BinaryMask getMask() {
                return ((LocationData) this.instance).getMask();
            }

            public Builder setMask(BinaryMask value) {
                copyOnWrite();
                ((LocationData) this.instance).setMask(value);
                return this;
            }

            public Builder setMask(BinaryMask.Builder builderForValue) {
                copyOnWrite();
                ((LocationData) this.instance).setMask(builderForValue.build());
                return this;
            }

            public Builder mergeMask(BinaryMask value) {
                copyOnWrite();
                ((LocationData) this.instance).mergeMask(value);
                return this;
            }

            public Builder clearMask() {
                copyOnWrite();
                ((LocationData) this.instance).clearMask();
                return this;
            }

            @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationDataOrBuilder
            public List<RelativeKeypoint> getRelativeKeypointsList() {
                return Collections.unmodifiableList(((LocationData) this.instance).getRelativeKeypointsList());
            }

            @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationDataOrBuilder
            public int getRelativeKeypointsCount() {
                return ((LocationData) this.instance).getRelativeKeypointsCount();
            }

            @Override // com.google.mediapipe.formats.proto.LocationDataProto.LocationDataOrBuilder
            public RelativeKeypoint getRelativeKeypoints(int index) {
                return ((LocationData) this.instance).getRelativeKeypoints(index);
            }

            public Builder setRelativeKeypoints(int index, RelativeKeypoint value) {
                copyOnWrite();
                ((LocationData) this.instance).setRelativeKeypoints(index, value);
                return this;
            }

            public Builder setRelativeKeypoints(int index, RelativeKeypoint.Builder builderForValue) {
                copyOnWrite();
                ((LocationData) this.instance).setRelativeKeypoints(index, builderForValue.build());
                return this;
            }

            public Builder addRelativeKeypoints(RelativeKeypoint value) {
                copyOnWrite();
                ((LocationData) this.instance).addRelativeKeypoints(value);
                return this;
            }

            public Builder addRelativeKeypoints(int index, RelativeKeypoint value) {
                copyOnWrite();
                ((LocationData) this.instance).addRelativeKeypoints(index, value);
                return this;
            }

            public Builder addRelativeKeypoints(RelativeKeypoint.Builder builderForValue) {
                copyOnWrite();
                ((LocationData) this.instance).addRelativeKeypoints(builderForValue.build());
                return this;
            }

            public Builder addRelativeKeypoints(int index, RelativeKeypoint.Builder builderForValue) {
                copyOnWrite();
                ((LocationData) this.instance).addRelativeKeypoints(index, builderForValue.build());
                return this;
            }

            public Builder addAllRelativeKeypoints(Iterable<? extends RelativeKeypoint> values) {
                copyOnWrite();
                ((LocationData) this.instance).addAllRelativeKeypoints(values);
                return this;
            }

            public Builder clearRelativeKeypoints() {
                copyOnWrite();
                ((LocationData) this.instance).clearRelativeKeypoints();
                return this;
            }

            public Builder removeRelativeKeypoints(int index) {
                copyOnWrite();
                ((LocationData) this.instance).removeRelativeKeypoints(index);
                return this;
            }
        }

        /* JADX INFO: Access modifiers changed from: protected */
        @Override // com.google.protobuf.GeneratedMessageLite
        public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
            switch (method) {
                case NEW_MUTABLE_INSTANCE:
                    return new LocationData();
                case NEW_BUILDER:
                    return new Builder();
                case BUILD_MESSAGE_INFO:
                    Object[] objects = {"bitField0_", "format_", Format.internalGetVerifier(), "boundingBox_", "relativeBoundingBox_", "mask_", "relativeKeypoints_", RelativeKeypoint.class};
                    return newMessageInfo(DEFAULT_INSTANCE, "\u0001\u0005��\u0001\u0001\u0005\u0005��\u0001\u0001\u0001\f��\u0002\t\u0001\u0003\t\u0002\u0004Љ\u0003\u0005\u001b", objects);
                case GET_DEFAULT_INSTANCE:
                    return DEFAULT_INSTANCE;
                case GET_PARSER:
                    Parser<LocationData> parser = PARSER;
                    if (parser == null) {
                        synchronized (LocationData.class) {
                            parser = PARSER;
                            if (parser == null) {
                                parser = new GeneratedMessageLite.DefaultInstanceBasedParser<>(DEFAULT_INSTANCE);
                                PARSER = parser;
                            }
                        }
                    }
                    return parser;
                case GET_MEMOIZED_IS_INITIALIZED:
                    return Byte.valueOf(this.memoizedIsInitialized);
                case SET_MEMOIZED_IS_INITIALIZED:
                    this.memoizedIsInitialized = (byte) (arg0 == null ? 0 : 1);
                    return null;
                default:
                    throw new UnsupportedOperationException();
            }
        }

        static {
            LocationData defaultInstance = new LocationData();
            DEFAULT_INSTANCE = defaultInstance;
            GeneratedMessageLite.registerDefaultInstance(LocationData.class, defaultInstance);
        }

        public static LocationData getDefaultInstance() {
            return DEFAULT_INSTANCE;
        }

        public static Parser<LocationData> parser() {
            return DEFAULT_INSTANCE.getParserForType();
        }
    }
}