package com.google.mediapipe.tracking;

import com.google.android.material.internal.StaticLayoutBuilderCompat;
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
/* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/MotionModelsProto.class */
public final class MotionModelsProto {

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/MotionModelsProto$AffineModelOrBuilder.class */
    public interface AffineModelOrBuilder extends MessageLiteOrBuilder {
        boolean hasDx();

        float getDx();

        boolean hasDy();

        float getDy();

        boolean hasA();

        float getA();

        boolean hasB();

        float getB();

        boolean hasC();

        float getC();

        boolean hasD();

        float getD();
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/MotionModelsProto$HomographyOrBuilder.class */
    public interface HomographyOrBuilder extends MessageLiteOrBuilder {
        boolean hasH00();

        float getH00();

        boolean hasH01();

        float getH01();

        boolean hasH02();

        float getH02();

        boolean hasH10();

        float getH10();

        boolean hasH11();

        float getH11();

        boolean hasH12();

        float getH12();

        boolean hasH20();

        float getH20();

        boolean hasH21();

        float getH21();
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/MotionModelsProto$LinearSimilarityModelOrBuilder.class */
    public interface LinearSimilarityModelOrBuilder extends MessageLiteOrBuilder {
        boolean hasDx();

        float getDx();

        boolean hasDy();

        float getDy();

        boolean hasA();

        float getA();

        boolean hasB();

        float getB();
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/MotionModelsProto$MixtureAffineOrBuilder.class */
    public interface MixtureAffineOrBuilder extends MessageLiteOrBuilder {
        List<AffineModel> getModelList();

        AffineModel getModel(int index);

        int getModelCount();
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/MotionModelsProto$MixtureHomographyOrBuilder.class */
    public interface MixtureHomographyOrBuilder extends MessageLiteOrBuilder {
        List<Homography> getModelList();

        Homography getModel(int index);

        int getModelCount();

        boolean hasDof();

        MixtureHomography.VariableDOF getDof();
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/MotionModelsProto$MixtureLinearSimilarityOrBuilder.class */
    public interface MixtureLinearSimilarityOrBuilder extends MessageLiteOrBuilder {
        List<LinearSimilarityModel> getModelList();

        LinearSimilarityModel getModel(int index);

        int getModelCount();
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/MotionModelsProto$SimilarityModelOrBuilder.class */
    public interface SimilarityModelOrBuilder extends MessageLiteOrBuilder {
        boolean hasDx();

        float getDx();

        boolean hasDy();

        float getDy();

        boolean hasScale();

        float getScale();

        boolean hasRotation();

        float getRotation();
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/MotionModelsProto$TranslationModelOrBuilder.class */
    public interface TranslationModelOrBuilder extends MessageLiteOrBuilder {
        boolean hasDx();

        float getDx();

        boolean hasDy();

        float getDy();
    }

    private MotionModelsProto() {
    }

    public static void registerAllExtensions(ExtensionRegistryLite registry) {
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/MotionModelsProto$TranslationModel.class */
    public static final class TranslationModel extends GeneratedMessageLite<TranslationModel, Builder> implements TranslationModelOrBuilder {
        private int bitField0_;
        public static final int DX_FIELD_NUMBER = 1;
        private float dx_;
        public static final int DY_FIELD_NUMBER = 2;
        private float dy_;
        private static final TranslationModel DEFAULT_INSTANCE;
        private static volatile Parser<TranslationModel> PARSER;

        private TranslationModel() {
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.TranslationModelOrBuilder
        public boolean hasDx() {
            return (this.bitField0_ & 1) != 0;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.TranslationModelOrBuilder
        public float getDx() {
            return this.dx_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setDx(float value) {
            this.bitField0_ |= 1;
            this.dx_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearDx() {
            this.bitField0_ &= -2;
            this.dx_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.TranslationModelOrBuilder
        public boolean hasDy() {
            return (this.bitField0_ & 2) != 0;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.TranslationModelOrBuilder
        public float getDy() {
            return this.dy_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setDy(float value) {
            this.bitField0_ |= 2;
            this.dy_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearDy() {
            this.bitField0_ &= -3;
            this.dy_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        public static TranslationModel parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
            return (TranslationModel) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static TranslationModel parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (TranslationModel) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static TranslationModel parseFrom(ByteString data) throws InvalidProtocolBufferException {
            return (TranslationModel) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static TranslationModel parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (TranslationModel) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static TranslationModel parseFrom(byte[] data) throws InvalidProtocolBufferException {
            return (TranslationModel) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static TranslationModel parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (TranslationModel) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static TranslationModel parseFrom(InputStream input) throws IOException {
            return (TranslationModel) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static TranslationModel parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (TranslationModel) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static TranslationModel parseDelimitedFrom(InputStream input) throws IOException {
            return (TranslationModel) parseDelimitedFrom(DEFAULT_INSTANCE, input);
        }

        public static TranslationModel parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (TranslationModel) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static TranslationModel parseFrom(CodedInputStream input) throws IOException {
            return (TranslationModel) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static TranslationModel parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (TranslationModel) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Builder newBuilder() {
            return DEFAULT_INSTANCE.createBuilder();
        }

        public static Builder newBuilder(TranslationModel prototype) {
            return DEFAULT_INSTANCE.createBuilder(prototype);
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/MotionModelsProto$TranslationModel$Builder.class */
        public static final class Builder extends GeneratedMessageLite.Builder<TranslationModel, Builder> implements TranslationModelOrBuilder {
            private Builder() {
                super(TranslationModel.DEFAULT_INSTANCE);
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.TranslationModelOrBuilder
            public boolean hasDx() {
                return ((TranslationModel) this.instance).hasDx();
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.TranslationModelOrBuilder
            public float getDx() {
                return ((TranslationModel) this.instance).getDx();
            }

            public Builder setDx(float value) {
                copyOnWrite();
                ((TranslationModel) this.instance).setDx(value);
                return this;
            }

            public Builder clearDx() {
                copyOnWrite();
                ((TranslationModel) this.instance).clearDx();
                return this;
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.TranslationModelOrBuilder
            public boolean hasDy() {
                return ((TranslationModel) this.instance).hasDy();
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.TranslationModelOrBuilder
            public float getDy() {
                return ((TranslationModel) this.instance).getDy();
            }

            public Builder setDy(float value) {
                copyOnWrite();
                ((TranslationModel) this.instance).setDy(value);
                return this;
            }

            public Builder clearDy() {
                copyOnWrite();
                ((TranslationModel) this.instance).clearDy();
                return this;
            }
        }

        /* JADX INFO: Access modifiers changed from: protected */
        @Override // com.google.protobuf.GeneratedMessageLite
        public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
            switch (method) {
                case NEW_MUTABLE_INSTANCE:
                    return new TranslationModel();
                case NEW_BUILDER:
                    return new Builder();
                case BUILD_MESSAGE_INFO:
                    Object[] objects = {"bitField0_", "dx_", "dy_"};
                    return newMessageInfo(DEFAULT_INSTANCE, "\u0001\u0002��\u0001\u0001\u0002\u0002������\u0001\u0001��\u0002\u0001\u0001", objects);
                case GET_DEFAULT_INSTANCE:
                    return DEFAULT_INSTANCE;
                case GET_PARSER:
                    Parser<TranslationModel> parser = PARSER;
                    if (parser == null) {
                        synchronized (TranslationModel.class) {
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
            TranslationModel defaultInstance = new TranslationModel();
            DEFAULT_INSTANCE = defaultInstance;
            GeneratedMessageLite.registerDefaultInstance(TranslationModel.class, defaultInstance);
        }

        public static TranslationModel getDefaultInstance() {
            return DEFAULT_INSTANCE;
        }

        public static Parser<TranslationModel> parser() {
            return DEFAULT_INSTANCE.getParserForType();
        }
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/MotionModelsProto$SimilarityModel.class */
    public static final class SimilarityModel extends GeneratedMessageLite<SimilarityModel, Builder> implements SimilarityModelOrBuilder {
        private int bitField0_;
        public static final int DX_FIELD_NUMBER = 1;
        private float dx_;
        public static final int DY_FIELD_NUMBER = 2;
        private float dy_;
        public static final int SCALE_FIELD_NUMBER = 3;
        private float scale_ = 1.0f;
        public static final int ROTATION_FIELD_NUMBER = 4;
        private float rotation_;
        private static final SimilarityModel DEFAULT_INSTANCE;
        private static volatile Parser<SimilarityModel> PARSER;

        private SimilarityModel() {
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.SimilarityModelOrBuilder
        public boolean hasDx() {
            return (this.bitField0_ & 1) != 0;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.SimilarityModelOrBuilder
        public float getDx() {
            return this.dx_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setDx(float value) {
            this.bitField0_ |= 1;
            this.dx_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearDx() {
            this.bitField0_ &= -2;
            this.dx_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.SimilarityModelOrBuilder
        public boolean hasDy() {
            return (this.bitField0_ & 2) != 0;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.SimilarityModelOrBuilder
        public float getDy() {
            return this.dy_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setDy(float value) {
            this.bitField0_ |= 2;
            this.dy_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearDy() {
            this.bitField0_ &= -3;
            this.dy_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.SimilarityModelOrBuilder
        public boolean hasScale() {
            return (this.bitField0_ & 4) != 0;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.SimilarityModelOrBuilder
        public float getScale() {
            return this.scale_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setScale(float value) {
            this.bitField0_ |= 4;
            this.scale_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearScale() {
            this.bitField0_ &= -5;
            this.scale_ = 1.0f;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.SimilarityModelOrBuilder
        public boolean hasRotation() {
            return (this.bitField0_ & 8) != 0;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.SimilarityModelOrBuilder
        public float getRotation() {
            return this.rotation_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setRotation(float value) {
            this.bitField0_ |= 8;
            this.rotation_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearRotation() {
            this.bitField0_ &= -9;
            this.rotation_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        public static SimilarityModel parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
            return (SimilarityModel) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static SimilarityModel parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (SimilarityModel) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static SimilarityModel parseFrom(ByteString data) throws InvalidProtocolBufferException {
            return (SimilarityModel) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static SimilarityModel parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (SimilarityModel) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static SimilarityModel parseFrom(byte[] data) throws InvalidProtocolBufferException {
            return (SimilarityModel) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static SimilarityModel parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (SimilarityModel) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static SimilarityModel parseFrom(InputStream input) throws IOException {
            return (SimilarityModel) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static SimilarityModel parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (SimilarityModel) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static SimilarityModel parseDelimitedFrom(InputStream input) throws IOException {
            return (SimilarityModel) parseDelimitedFrom(DEFAULT_INSTANCE, input);
        }

        public static SimilarityModel parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (SimilarityModel) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static SimilarityModel parseFrom(CodedInputStream input) throws IOException {
            return (SimilarityModel) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static SimilarityModel parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (SimilarityModel) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Builder newBuilder() {
            return DEFAULT_INSTANCE.createBuilder();
        }

        public static Builder newBuilder(SimilarityModel prototype) {
            return DEFAULT_INSTANCE.createBuilder(prototype);
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/MotionModelsProto$SimilarityModel$Builder.class */
        public static final class Builder extends GeneratedMessageLite.Builder<SimilarityModel, Builder> implements SimilarityModelOrBuilder {
            private Builder() {
                super(SimilarityModel.DEFAULT_INSTANCE);
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.SimilarityModelOrBuilder
            public boolean hasDx() {
                return ((SimilarityModel) this.instance).hasDx();
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.SimilarityModelOrBuilder
            public float getDx() {
                return ((SimilarityModel) this.instance).getDx();
            }

            public Builder setDx(float value) {
                copyOnWrite();
                ((SimilarityModel) this.instance).setDx(value);
                return this;
            }

            public Builder clearDx() {
                copyOnWrite();
                ((SimilarityModel) this.instance).clearDx();
                return this;
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.SimilarityModelOrBuilder
            public boolean hasDy() {
                return ((SimilarityModel) this.instance).hasDy();
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.SimilarityModelOrBuilder
            public float getDy() {
                return ((SimilarityModel) this.instance).getDy();
            }

            public Builder setDy(float value) {
                copyOnWrite();
                ((SimilarityModel) this.instance).setDy(value);
                return this;
            }

            public Builder clearDy() {
                copyOnWrite();
                ((SimilarityModel) this.instance).clearDy();
                return this;
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.SimilarityModelOrBuilder
            public boolean hasScale() {
                return ((SimilarityModel) this.instance).hasScale();
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.SimilarityModelOrBuilder
            public float getScale() {
                return ((SimilarityModel) this.instance).getScale();
            }

            public Builder setScale(float value) {
                copyOnWrite();
                ((SimilarityModel) this.instance).setScale(value);
                return this;
            }

            public Builder clearScale() {
                copyOnWrite();
                ((SimilarityModel) this.instance).clearScale();
                return this;
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.SimilarityModelOrBuilder
            public boolean hasRotation() {
                return ((SimilarityModel) this.instance).hasRotation();
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.SimilarityModelOrBuilder
            public float getRotation() {
                return ((SimilarityModel) this.instance).getRotation();
            }

            public Builder setRotation(float value) {
                copyOnWrite();
                ((SimilarityModel) this.instance).setRotation(value);
                return this;
            }

            public Builder clearRotation() {
                copyOnWrite();
                ((SimilarityModel) this.instance).clearRotation();
                return this;
            }
        }

        /* JADX INFO: Access modifiers changed from: protected */
        @Override // com.google.protobuf.GeneratedMessageLite
        public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
            switch (method) {
                case NEW_MUTABLE_INSTANCE:
                    return new SimilarityModel();
                case NEW_BUILDER:
                    return new Builder();
                case BUILD_MESSAGE_INFO:
                    Object[] objects = {"bitField0_", "dx_", "dy_", "scale_", "rotation_"};
                    return newMessageInfo(DEFAULT_INSTANCE, "\u0001\u0004��\u0001\u0001\u0004\u0004������\u0001\u0001��\u0002\u0001\u0001\u0003\u0001\u0002\u0004\u0001\u0003", objects);
                case GET_DEFAULT_INSTANCE:
                    return DEFAULT_INSTANCE;
                case GET_PARSER:
                    Parser<SimilarityModel> parser = PARSER;
                    if (parser == null) {
                        synchronized (SimilarityModel.class) {
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
            SimilarityModel defaultInstance = new SimilarityModel();
            DEFAULT_INSTANCE = defaultInstance;
            GeneratedMessageLite.registerDefaultInstance(SimilarityModel.class, defaultInstance);
        }

        public static SimilarityModel getDefaultInstance() {
            return DEFAULT_INSTANCE;
        }

        public static Parser<SimilarityModel> parser() {
            return DEFAULT_INSTANCE.getParserForType();
        }
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/MotionModelsProto$LinearSimilarityModel.class */
    public static final class LinearSimilarityModel extends GeneratedMessageLite<LinearSimilarityModel, Builder> implements LinearSimilarityModelOrBuilder {
        private int bitField0_;
        public static final int DX_FIELD_NUMBER = 1;
        private float dx_;
        public static final int DY_FIELD_NUMBER = 2;
        private float dy_;
        public static final int A_FIELD_NUMBER = 3;
        private float a_ = 1.0f;
        public static final int B_FIELD_NUMBER = 4;
        private float b_;
        private static final LinearSimilarityModel DEFAULT_INSTANCE;
        private static volatile Parser<LinearSimilarityModel> PARSER;

        private LinearSimilarityModel() {
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.LinearSimilarityModelOrBuilder
        public boolean hasDx() {
            return (this.bitField0_ & 1) != 0;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.LinearSimilarityModelOrBuilder
        public float getDx() {
            return this.dx_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setDx(float value) {
            this.bitField0_ |= 1;
            this.dx_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearDx() {
            this.bitField0_ &= -2;
            this.dx_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.LinearSimilarityModelOrBuilder
        public boolean hasDy() {
            return (this.bitField0_ & 2) != 0;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.LinearSimilarityModelOrBuilder
        public float getDy() {
            return this.dy_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setDy(float value) {
            this.bitField0_ |= 2;
            this.dy_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearDy() {
            this.bitField0_ &= -3;
            this.dy_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.LinearSimilarityModelOrBuilder
        public boolean hasA() {
            return (this.bitField0_ & 4) != 0;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.LinearSimilarityModelOrBuilder
        public float getA() {
            return this.a_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setA(float value) {
            this.bitField0_ |= 4;
            this.a_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearA() {
            this.bitField0_ &= -5;
            this.a_ = 1.0f;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.LinearSimilarityModelOrBuilder
        public boolean hasB() {
            return (this.bitField0_ & 8) != 0;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.LinearSimilarityModelOrBuilder
        public float getB() {
            return this.b_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setB(float value) {
            this.bitField0_ |= 8;
            this.b_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearB() {
            this.bitField0_ &= -9;
            this.b_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        public static LinearSimilarityModel parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
            return (LinearSimilarityModel) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static LinearSimilarityModel parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (LinearSimilarityModel) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static LinearSimilarityModel parseFrom(ByteString data) throws InvalidProtocolBufferException {
            return (LinearSimilarityModel) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static LinearSimilarityModel parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (LinearSimilarityModel) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static LinearSimilarityModel parseFrom(byte[] data) throws InvalidProtocolBufferException {
            return (LinearSimilarityModel) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static LinearSimilarityModel parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (LinearSimilarityModel) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static LinearSimilarityModel parseFrom(InputStream input) throws IOException {
            return (LinearSimilarityModel) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static LinearSimilarityModel parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (LinearSimilarityModel) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static LinearSimilarityModel parseDelimitedFrom(InputStream input) throws IOException {
            return (LinearSimilarityModel) parseDelimitedFrom(DEFAULT_INSTANCE, input);
        }

        public static LinearSimilarityModel parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (LinearSimilarityModel) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static LinearSimilarityModel parseFrom(CodedInputStream input) throws IOException {
            return (LinearSimilarityModel) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static LinearSimilarityModel parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (LinearSimilarityModel) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Builder newBuilder() {
            return DEFAULT_INSTANCE.createBuilder();
        }

        public static Builder newBuilder(LinearSimilarityModel prototype) {
            return DEFAULT_INSTANCE.createBuilder(prototype);
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/MotionModelsProto$LinearSimilarityModel$Builder.class */
        public static final class Builder extends GeneratedMessageLite.Builder<LinearSimilarityModel, Builder> implements LinearSimilarityModelOrBuilder {
            private Builder() {
                super(LinearSimilarityModel.DEFAULT_INSTANCE);
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.LinearSimilarityModelOrBuilder
            public boolean hasDx() {
                return ((LinearSimilarityModel) this.instance).hasDx();
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.LinearSimilarityModelOrBuilder
            public float getDx() {
                return ((LinearSimilarityModel) this.instance).getDx();
            }

            public Builder setDx(float value) {
                copyOnWrite();
                ((LinearSimilarityModel) this.instance).setDx(value);
                return this;
            }

            public Builder clearDx() {
                copyOnWrite();
                ((LinearSimilarityModel) this.instance).clearDx();
                return this;
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.LinearSimilarityModelOrBuilder
            public boolean hasDy() {
                return ((LinearSimilarityModel) this.instance).hasDy();
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.LinearSimilarityModelOrBuilder
            public float getDy() {
                return ((LinearSimilarityModel) this.instance).getDy();
            }

            public Builder setDy(float value) {
                copyOnWrite();
                ((LinearSimilarityModel) this.instance).setDy(value);
                return this;
            }

            public Builder clearDy() {
                copyOnWrite();
                ((LinearSimilarityModel) this.instance).clearDy();
                return this;
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.LinearSimilarityModelOrBuilder
            public boolean hasA() {
                return ((LinearSimilarityModel) this.instance).hasA();
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.LinearSimilarityModelOrBuilder
            public float getA() {
                return ((LinearSimilarityModel) this.instance).getA();
            }

            public Builder setA(float value) {
                copyOnWrite();
                ((LinearSimilarityModel) this.instance).setA(value);
                return this;
            }

            public Builder clearA() {
                copyOnWrite();
                ((LinearSimilarityModel) this.instance).clearA();
                return this;
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.LinearSimilarityModelOrBuilder
            public boolean hasB() {
                return ((LinearSimilarityModel) this.instance).hasB();
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.LinearSimilarityModelOrBuilder
            public float getB() {
                return ((LinearSimilarityModel) this.instance).getB();
            }

            public Builder setB(float value) {
                copyOnWrite();
                ((LinearSimilarityModel) this.instance).setB(value);
                return this;
            }

            public Builder clearB() {
                copyOnWrite();
                ((LinearSimilarityModel) this.instance).clearB();
                return this;
            }
        }

        /* JADX INFO: Access modifiers changed from: protected */
        @Override // com.google.protobuf.GeneratedMessageLite
        public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
            switch (method) {
                case NEW_MUTABLE_INSTANCE:
                    return new LinearSimilarityModel();
                case NEW_BUILDER:
                    return new Builder();
                case BUILD_MESSAGE_INFO:
                    Object[] objects = {"bitField0_", "dx_", "dy_", "a_", "b_"};
                    return newMessageInfo(DEFAULT_INSTANCE, "\u0001\u0004��\u0001\u0001\u0004\u0004������\u0001\u0001��\u0002\u0001\u0001\u0003\u0001\u0002\u0004\u0001\u0003", objects);
                case GET_DEFAULT_INSTANCE:
                    return DEFAULT_INSTANCE;
                case GET_PARSER:
                    Parser<LinearSimilarityModel> parser = PARSER;
                    if (parser == null) {
                        synchronized (LinearSimilarityModel.class) {
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
            LinearSimilarityModel defaultInstance = new LinearSimilarityModel();
            DEFAULT_INSTANCE = defaultInstance;
            GeneratedMessageLite.registerDefaultInstance(LinearSimilarityModel.class, defaultInstance);
        }

        public static LinearSimilarityModel getDefaultInstance() {
            return DEFAULT_INSTANCE;
        }

        public static Parser<LinearSimilarityModel> parser() {
            return DEFAULT_INSTANCE.getParserForType();
        }
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/MotionModelsProto$AffineModel.class */
    public static final class AffineModel extends GeneratedMessageLite<AffineModel, Builder> implements AffineModelOrBuilder {
        private int bitField0_;
        public static final int DX_FIELD_NUMBER = 1;
        private float dx_;
        public static final int DY_FIELD_NUMBER = 2;
        private float dy_;
        public static final int A_FIELD_NUMBER = 3;
        public static final int B_FIELD_NUMBER = 4;
        private float b_;
        public static final int C_FIELD_NUMBER = 5;
        private float c_;
        public static final int D_FIELD_NUMBER = 6;
        private static final AffineModel DEFAULT_INSTANCE;
        private static volatile Parser<AffineModel> PARSER;
        private float a_ = 1.0f;
        private float d_ = 1.0f;

        private AffineModel() {
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.AffineModelOrBuilder
        public boolean hasDx() {
            return (this.bitField0_ & 1) != 0;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.AffineModelOrBuilder
        public float getDx() {
            return this.dx_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setDx(float value) {
            this.bitField0_ |= 1;
            this.dx_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearDx() {
            this.bitField0_ &= -2;
            this.dx_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.AffineModelOrBuilder
        public boolean hasDy() {
            return (this.bitField0_ & 2) != 0;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.AffineModelOrBuilder
        public float getDy() {
            return this.dy_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setDy(float value) {
            this.bitField0_ |= 2;
            this.dy_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearDy() {
            this.bitField0_ &= -3;
            this.dy_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.AffineModelOrBuilder
        public boolean hasA() {
            return (this.bitField0_ & 4) != 0;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.AffineModelOrBuilder
        public float getA() {
            return this.a_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setA(float value) {
            this.bitField0_ |= 4;
            this.a_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearA() {
            this.bitField0_ &= -5;
            this.a_ = 1.0f;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.AffineModelOrBuilder
        public boolean hasB() {
            return (this.bitField0_ & 8) != 0;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.AffineModelOrBuilder
        public float getB() {
            return this.b_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setB(float value) {
            this.bitField0_ |= 8;
            this.b_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearB() {
            this.bitField0_ &= -9;
            this.b_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.AffineModelOrBuilder
        public boolean hasC() {
            return (this.bitField0_ & 16) != 0;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.AffineModelOrBuilder
        public float getC() {
            return this.c_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setC(float value) {
            this.bitField0_ |= 16;
            this.c_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearC() {
            this.bitField0_ &= -17;
            this.c_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.AffineModelOrBuilder
        public boolean hasD() {
            return (this.bitField0_ & 32) != 0;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.AffineModelOrBuilder
        public float getD() {
            return this.d_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setD(float value) {
            this.bitField0_ |= 32;
            this.d_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearD() {
            this.bitField0_ &= -33;
            this.d_ = 1.0f;
        }

        public static AffineModel parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
            return (AffineModel) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static AffineModel parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (AffineModel) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static AffineModel parseFrom(ByteString data) throws InvalidProtocolBufferException {
            return (AffineModel) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static AffineModel parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (AffineModel) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static AffineModel parseFrom(byte[] data) throws InvalidProtocolBufferException {
            return (AffineModel) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static AffineModel parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (AffineModel) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static AffineModel parseFrom(InputStream input) throws IOException {
            return (AffineModel) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static AffineModel parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (AffineModel) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static AffineModel parseDelimitedFrom(InputStream input) throws IOException {
            return (AffineModel) parseDelimitedFrom(DEFAULT_INSTANCE, input);
        }

        public static AffineModel parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (AffineModel) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static AffineModel parseFrom(CodedInputStream input) throws IOException {
            return (AffineModel) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static AffineModel parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (AffineModel) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Builder newBuilder() {
            return DEFAULT_INSTANCE.createBuilder();
        }

        public static Builder newBuilder(AffineModel prototype) {
            return DEFAULT_INSTANCE.createBuilder(prototype);
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/MotionModelsProto$AffineModel$Builder.class */
        public static final class Builder extends GeneratedMessageLite.Builder<AffineModel, Builder> implements AffineModelOrBuilder {
            private Builder() {
                super(AffineModel.DEFAULT_INSTANCE);
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.AffineModelOrBuilder
            public boolean hasDx() {
                return ((AffineModel) this.instance).hasDx();
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.AffineModelOrBuilder
            public float getDx() {
                return ((AffineModel) this.instance).getDx();
            }

            public Builder setDx(float value) {
                copyOnWrite();
                ((AffineModel) this.instance).setDx(value);
                return this;
            }

            public Builder clearDx() {
                copyOnWrite();
                ((AffineModel) this.instance).clearDx();
                return this;
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.AffineModelOrBuilder
            public boolean hasDy() {
                return ((AffineModel) this.instance).hasDy();
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.AffineModelOrBuilder
            public float getDy() {
                return ((AffineModel) this.instance).getDy();
            }

            public Builder setDy(float value) {
                copyOnWrite();
                ((AffineModel) this.instance).setDy(value);
                return this;
            }

            public Builder clearDy() {
                copyOnWrite();
                ((AffineModel) this.instance).clearDy();
                return this;
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.AffineModelOrBuilder
            public boolean hasA() {
                return ((AffineModel) this.instance).hasA();
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.AffineModelOrBuilder
            public float getA() {
                return ((AffineModel) this.instance).getA();
            }

            public Builder setA(float value) {
                copyOnWrite();
                ((AffineModel) this.instance).setA(value);
                return this;
            }

            public Builder clearA() {
                copyOnWrite();
                ((AffineModel) this.instance).clearA();
                return this;
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.AffineModelOrBuilder
            public boolean hasB() {
                return ((AffineModel) this.instance).hasB();
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.AffineModelOrBuilder
            public float getB() {
                return ((AffineModel) this.instance).getB();
            }

            public Builder setB(float value) {
                copyOnWrite();
                ((AffineModel) this.instance).setB(value);
                return this;
            }

            public Builder clearB() {
                copyOnWrite();
                ((AffineModel) this.instance).clearB();
                return this;
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.AffineModelOrBuilder
            public boolean hasC() {
                return ((AffineModel) this.instance).hasC();
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.AffineModelOrBuilder
            public float getC() {
                return ((AffineModel) this.instance).getC();
            }

            public Builder setC(float value) {
                copyOnWrite();
                ((AffineModel) this.instance).setC(value);
                return this;
            }

            public Builder clearC() {
                copyOnWrite();
                ((AffineModel) this.instance).clearC();
                return this;
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.AffineModelOrBuilder
            public boolean hasD() {
                return ((AffineModel) this.instance).hasD();
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.AffineModelOrBuilder
            public float getD() {
                return ((AffineModel) this.instance).getD();
            }

            public Builder setD(float value) {
                copyOnWrite();
                ((AffineModel) this.instance).setD(value);
                return this;
            }

            public Builder clearD() {
                copyOnWrite();
                ((AffineModel) this.instance).clearD();
                return this;
            }
        }

        /* JADX INFO: Access modifiers changed from: protected */
        @Override // com.google.protobuf.GeneratedMessageLite
        public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
            switch (method) {
                case NEW_MUTABLE_INSTANCE:
                    return new AffineModel();
                case NEW_BUILDER:
                    return new Builder();
                case BUILD_MESSAGE_INFO:
                    Object[] objects = {"bitField0_", "dx_", "dy_", "a_", "b_", "c_", "d_"};
                    return newMessageInfo(DEFAULT_INSTANCE, "\u0001\u0006��\u0001\u0001\u0006\u0006������\u0001\u0001��\u0002\u0001\u0001\u0003\u0001\u0002\u0004\u0001\u0003\u0005\u0001\u0004\u0006\u0001\u0005", objects);
                case GET_DEFAULT_INSTANCE:
                    return DEFAULT_INSTANCE;
                case GET_PARSER:
                    Parser<AffineModel> parser = PARSER;
                    if (parser == null) {
                        synchronized (AffineModel.class) {
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
            AffineModel defaultInstance = new AffineModel();
            DEFAULT_INSTANCE = defaultInstance;
            GeneratedMessageLite.registerDefaultInstance(AffineModel.class, defaultInstance);
        }

        public static AffineModel getDefaultInstance() {
            return DEFAULT_INSTANCE;
        }

        public static Parser<AffineModel> parser() {
            return DEFAULT_INSTANCE.getParserForType();
        }
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/MotionModelsProto$Homography.class */
    public static final class Homography extends GeneratedMessageLite<Homography, Builder> implements HomographyOrBuilder {
        private int bitField0_;
        public static final int H_00_FIELD_NUMBER = 1;
        public static final int H_01_FIELD_NUMBER = 2;
        private float h01_;
        public static final int H_02_FIELD_NUMBER = 3;
        private float h02_;
        public static final int H_10_FIELD_NUMBER = 4;
        private float h10_;
        public static final int H_11_FIELD_NUMBER = 5;
        public static final int H_12_FIELD_NUMBER = 6;
        private float h12_;
        public static final int H_20_FIELD_NUMBER = 7;
        private float h20_;
        public static final int H_21_FIELD_NUMBER = 8;
        private float h21_;
        private static final Homography DEFAULT_INSTANCE;
        private static volatile Parser<Homography> PARSER;
        private float h00_ = 1.0f;
        private float h11_ = 1.0f;

        private Homography() {
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.HomographyOrBuilder
        public boolean hasH00() {
            return (this.bitField0_ & 1) != 0;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.HomographyOrBuilder
        public float getH00() {
            return this.h00_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setH00(float value) {
            this.bitField0_ |= 1;
            this.h00_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearH00() {
            this.bitField0_ &= -2;
            this.h00_ = 1.0f;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.HomographyOrBuilder
        public boolean hasH01() {
            return (this.bitField0_ & 2) != 0;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.HomographyOrBuilder
        public float getH01() {
            return this.h01_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setH01(float value) {
            this.bitField0_ |= 2;
            this.h01_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearH01() {
            this.bitField0_ &= -3;
            this.h01_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.HomographyOrBuilder
        public boolean hasH02() {
            return (this.bitField0_ & 4) != 0;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.HomographyOrBuilder
        public float getH02() {
            return this.h02_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setH02(float value) {
            this.bitField0_ |= 4;
            this.h02_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearH02() {
            this.bitField0_ &= -5;
            this.h02_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.HomographyOrBuilder
        public boolean hasH10() {
            return (this.bitField0_ & 8) != 0;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.HomographyOrBuilder
        public float getH10() {
            return this.h10_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setH10(float value) {
            this.bitField0_ |= 8;
            this.h10_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearH10() {
            this.bitField0_ &= -9;
            this.h10_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.HomographyOrBuilder
        public boolean hasH11() {
            return (this.bitField0_ & 16) != 0;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.HomographyOrBuilder
        public float getH11() {
            return this.h11_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setH11(float value) {
            this.bitField0_ |= 16;
            this.h11_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearH11() {
            this.bitField0_ &= -17;
            this.h11_ = 1.0f;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.HomographyOrBuilder
        public boolean hasH12() {
            return (this.bitField0_ & 32) != 0;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.HomographyOrBuilder
        public float getH12() {
            return this.h12_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setH12(float value) {
            this.bitField0_ |= 32;
            this.h12_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearH12() {
            this.bitField0_ &= -33;
            this.h12_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.HomographyOrBuilder
        public boolean hasH20() {
            return (this.bitField0_ & 64) != 0;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.HomographyOrBuilder
        public float getH20() {
            return this.h20_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setH20(float value) {
            this.bitField0_ |= 64;
            this.h20_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearH20() {
            this.bitField0_ &= -65;
            this.h20_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.HomographyOrBuilder
        public boolean hasH21() {
            return (this.bitField0_ & 128) != 0;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.HomographyOrBuilder
        public float getH21() {
            return this.h21_;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setH21(float value) {
            this.bitField0_ |= 128;
            this.h21_ = value;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearH21() {
            this.bitField0_ &= -129;
            this.h21_ = StaticLayoutBuilderCompat.DEFAULT_LINE_SPACING_ADD;
        }

        public static Homography parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
            return (Homography) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static Homography parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (Homography) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static Homography parseFrom(ByteString data) throws InvalidProtocolBufferException {
            return (Homography) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static Homography parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (Homography) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static Homography parseFrom(byte[] data) throws InvalidProtocolBufferException {
            return (Homography) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static Homography parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (Homography) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static Homography parseFrom(InputStream input) throws IOException {
            return (Homography) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static Homography parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (Homography) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Homography parseDelimitedFrom(InputStream input) throws IOException {
            return (Homography) parseDelimitedFrom(DEFAULT_INSTANCE, input);
        }

        public static Homography parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (Homography) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Homography parseFrom(CodedInputStream input) throws IOException {
            return (Homography) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static Homography parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (Homography) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Builder newBuilder() {
            return DEFAULT_INSTANCE.createBuilder();
        }

        public static Builder newBuilder(Homography prototype) {
            return DEFAULT_INSTANCE.createBuilder(prototype);
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/MotionModelsProto$Homography$Builder.class */
        public static final class Builder extends GeneratedMessageLite.Builder<Homography, Builder> implements HomographyOrBuilder {
            private Builder() {
                super(Homography.DEFAULT_INSTANCE);
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.HomographyOrBuilder
            public boolean hasH00() {
                return ((Homography) this.instance).hasH00();
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.HomographyOrBuilder
            public float getH00() {
                return ((Homography) this.instance).getH00();
            }

            public Builder setH00(float value) {
                copyOnWrite();
                ((Homography) this.instance).setH00(value);
                return this;
            }

            public Builder clearH00() {
                copyOnWrite();
                ((Homography) this.instance).clearH00();
                return this;
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.HomographyOrBuilder
            public boolean hasH01() {
                return ((Homography) this.instance).hasH01();
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.HomographyOrBuilder
            public float getH01() {
                return ((Homography) this.instance).getH01();
            }

            public Builder setH01(float value) {
                copyOnWrite();
                ((Homography) this.instance).setH01(value);
                return this;
            }

            public Builder clearH01() {
                copyOnWrite();
                ((Homography) this.instance).clearH01();
                return this;
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.HomographyOrBuilder
            public boolean hasH02() {
                return ((Homography) this.instance).hasH02();
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.HomographyOrBuilder
            public float getH02() {
                return ((Homography) this.instance).getH02();
            }

            public Builder setH02(float value) {
                copyOnWrite();
                ((Homography) this.instance).setH02(value);
                return this;
            }

            public Builder clearH02() {
                copyOnWrite();
                ((Homography) this.instance).clearH02();
                return this;
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.HomographyOrBuilder
            public boolean hasH10() {
                return ((Homography) this.instance).hasH10();
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.HomographyOrBuilder
            public float getH10() {
                return ((Homography) this.instance).getH10();
            }

            public Builder setH10(float value) {
                copyOnWrite();
                ((Homography) this.instance).setH10(value);
                return this;
            }

            public Builder clearH10() {
                copyOnWrite();
                ((Homography) this.instance).clearH10();
                return this;
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.HomographyOrBuilder
            public boolean hasH11() {
                return ((Homography) this.instance).hasH11();
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.HomographyOrBuilder
            public float getH11() {
                return ((Homography) this.instance).getH11();
            }

            public Builder setH11(float value) {
                copyOnWrite();
                ((Homography) this.instance).setH11(value);
                return this;
            }

            public Builder clearH11() {
                copyOnWrite();
                ((Homography) this.instance).clearH11();
                return this;
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.HomographyOrBuilder
            public boolean hasH12() {
                return ((Homography) this.instance).hasH12();
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.HomographyOrBuilder
            public float getH12() {
                return ((Homography) this.instance).getH12();
            }

            public Builder setH12(float value) {
                copyOnWrite();
                ((Homography) this.instance).setH12(value);
                return this;
            }

            public Builder clearH12() {
                copyOnWrite();
                ((Homography) this.instance).clearH12();
                return this;
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.HomographyOrBuilder
            public boolean hasH20() {
                return ((Homography) this.instance).hasH20();
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.HomographyOrBuilder
            public float getH20() {
                return ((Homography) this.instance).getH20();
            }

            public Builder setH20(float value) {
                copyOnWrite();
                ((Homography) this.instance).setH20(value);
                return this;
            }

            public Builder clearH20() {
                copyOnWrite();
                ((Homography) this.instance).clearH20();
                return this;
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.HomographyOrBuilder
            public boolean hasH21() {
                return ((Homography) this.instance).hasH21();
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.HomographyOrBuilder
            public float getH21() {
                return ((Homography) this.instance).getH21();
            }

            public Builder setH21(float value) {
                copyOnWrite();
                ((Homography) this.instance).setH21(value);
                return this;
            }

            public Builder clearH21() {
                copyOnWrite();
                ((Homography) this.instance).clearH21();
                return this;
            }
        }

        /* JADX INFO: Access modifiers changed from: protected */
        @Override // com.google.protobuf.GeneratedMessageLite
        public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
            switch (method) {
                case NEW_MUTABLE_INSTANCE:
                    return new Homography();
                case NEW_BUILDER:
                    return new Builder();
                case BUILD_MESSAGE_INFO:
                    Object[] objects = {"bitField0_", "h00_", "h01_", "h02_", "h10_", "h11_", "h12_", "h20_", "h21_"};
                    return newMessageInfo(DEFAULT_INSTANCE, "\u0001\b��\u0001\u0001\b\b������\u0001\u0001��\u0002\u0001\u0001\u0003\u0001\u0002\u0004\u0001\u0003\u0005\u0001\u0004\u0006\u0001\u0005\u0007\u0001\u0006\b\u0001\u0007", objects);
                case GET_DEFAULT_INSTANCE:
                    return DEFAULT_INSTANCE;
                case GET_PARSER:
                    Parser<Homography> parser = PARSER;
                    if (parser == null) {
                        synchronized (Homography.class) {
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
            Homography defaultInstance = new Homography();
            DEFAULT_INSTANCE = defaultInstance;
            GeneratedMessageLite.registerDefaultInstance(Homography.class, defaultInstance);
        }

        public static Homography getDefaultInstance() {
            return DEFAULT_INSTANCE;
        }

        public static Parser<Homography> parser() {
            return DEFAULT_INSTANCE.getParserForType();
        }
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/MotionModelsProto$MixtureLinearSimilarity.class */
    public static final class MixtureLinearSimilarity extends GeneratedMessageLite<MixtureLinearSimilarity, Builder> implements MixtureLinearSimilarityOrBuilder {
        public static final int MODEL_FIELD_NUMBER = 1;
        private Internal.ProtobufList<LinearSimilarityModel> model_ = emptyProtobufList();
        private static final MixtureLinearSimilarity DEFAULT_INSTANCE;
        private static volatile Parser<MixtureLinearSimilarity> PARSER;

        private MixtureLinearSimilarity() {
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.MixtureLinearSimilarityOrBuilder
        public List<LinearSimilarityModel> getModelList() {
            return this.model_;
        }

        public List<? extends LinearSimilarityModelOrBuilder> getModelOrBuilderList() {
            return this.model_;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.MixtureLinearSimilarityOrBuilder
        public int getModelCount() {
            return this.model_.size();
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.MixtureLinearSimilarityOrBuilder
        public LinearSimilarityModel getModel(int index) {
            return this.model_.get(index);
        }

        public LinearSimilarityModelOrBuilder getModelOrBuilder(int index) {
            return this.model_.get(index);
        }

        private void ensureModelIsMutable() {
            if (!this.model_.isModifiable()) {
                this.model_ = GeneratedMessageLite.mutableCopy(this.model_);
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setModel(int index, LinearSimilarityModel value) {
            value.getClass();
            ensureModelIsMutable();
            this.model_.set(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addModel(LinearSimilarityModel value) {
            value.getClass();
            ensureModelIsMutable();
            this.model_.add(value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addModel(int index, LinearSimilarityModel value) {
            value.getClass();
            ensureModelIsMutable();
            this.model_.add(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addAllModel(Iterable<? extends LinearSimilarityModel> values) {
            ensureModelIsMutable();
            AbstractMessageLite.addAll((Iterable) values, (List) this.model_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearModel() {
            this.model_ = emptyProtobufList();
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void removeModel(int index) {
            ensureModelIsMutable();
            this.model_.remove(index);
        }

        public static MixtureLinearSimilarity parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
            return (MixtureLinearSimilarity) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static MixtureLinearSimilarity parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (MixtureLinearSimilarity) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static MixtureLinearSimilarity parseFrom(ByteString data) throws InvalidProtocolBufferException {
            return (MixtureLinearSimilarity) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static MixtureLinearSimilarity parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (MixtureLinearSimilarity) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static MixtureLinearSimilarity parseFrom(byte[] data) throws InvalidProtocolBufferException {
            return (MixtureLinearSimilarity) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static MixtureLinearSimilarity parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (MixtureLinearSimilarity) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static MixtureLinearSimilarity parseFrom(InputStream input) throws IOException {
            return (MixtureLinearSimilarity) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static MixtureLinearSimilarity parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (MixtureLinearSimilarity) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static MixtureLinearSimilarity parseDelimitedFrom(InputStream input) throws IOException {
            return (MixtureLinearSimilarity) parseDelimitedFrom(DEFAULT_INSTANCE, input);
        }

        public static MixtureLinearSimilarity parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (MixtureLinearSimilarity) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static MixtureLinearSimilarity parseFrom(CodedInputStream input) throws IOException {
            return (MixtureLinearSimilarity) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static MixtureLinearSimilarity parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (MixtureLinearSimilarity) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Builder newBuilder() {
            return DEFAULT_INSTANCE.createBuilder();
        }

        public static Builder newBuilder(MixtureLinearSimilarity prototype) {
            return DEFAULT_INSTANCE.createBuilder(prototype);
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/MotionModelsProto$MixtureLinearSimilarity$Builder.class */
        public static final class Builder extends GeneratedMessageLite.Builder<MixtureLinearSimilarity, Builder> implements MixtureLinearSimilarityOrBuilder {
            private Builder() {
                super(MixtureLinearSimilarity.DEFAULT_INSTANCE);
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.MixtureLinearSimilarityOrBuilder
            public List<LinearSimilarityModel> getModelList() {
                return Collections.unmodifiableList(((MixtureLinearSimilarity) this.instance).getModelList());
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.MixtureLinearSimilarityOrBuilder
            public int getModelCount() {
                return ((MixtureLinearSimilarity) this.instance).getModelCount();
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.MixtureLinearSimilarityOrBuilder
            public LinearSimilarityModel getModel(int index) {
                return ((MixtureLinearSimilarity) this.instance).getModel(index);
            }

            public Builder setModel(int index, LinearSimilarityModel value) {
                copyOnWrite();
                ((MixtureLinearSimilarity) this.instance).setModel(index, value);
                return this;
            }

            public Builder setModel(int index, LinearSimilarityModel.Builder builderForValue) {
                copyOnWrite();
                ((MixtureLinearSimilarity) this.instance).setModel(index, builderForValue.build());
                return this;
            }

            public Builder addModel(LinearSimilarityModel value) {
                copyOnWrite();
                ((MixtureLinearSimilarity) this.instance).addModel(value);
                return this;
            }

            public Builder addModel(int index, LinearSimilarityModel value) {
                copyOnWrite();
                ((MixtureLinearSimilarity) this.instance).addModel(index, value);
                return this;
            }

            public Builder addModel(LinearSimilarityModel.Builder builderForValue) {
                copyOnWrite();
                ((MixtureLinearSimilarity) this.instance).addModel(builderForValue.build());
                return this;
            }

            public Builder addModel(int index, LinearSimilarityModel.Builder builderForValue) {
                copyOnWrite();
                ((MixtureLinearSimilarity) this.instance).addModel(index, builderForValue.build());
                return this;
            }

            public Builder addAllModel(Iterable<? extends LinearSimilarityModel> values) {
                copyOnWrite();
                ((MixtureLinearSimilarity) this.instance).addAllModel(values);
                return this;
            }

            public Builder clearModel() {
                copyOnWrite();
                ((MixtureLinearSimilarity) this.instance).clearModel();
                return this;
            }

            public Builder removeModel(int index) {
                copyOnWrite();
                ((MixtureLinearSimilarity) this.instance).removeModel(index);
                return this;
            }
        }

        /* JADX INFO: Access modifiers changed from: protected */
        @Override // com.google.protobuf.GeneratedMessageLite
        public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
            switch (method) {
                case NEW_MUTABLE_INSTANCE:
                    return new MixtureLinearSimilarity();
                case NEW_BUILDER:
                    return new Builder();
                case BUILD_MESSAGE_INFO:
                    Object[] objects = {"model_", LinearSimilarityModel.class};
                    return newMessageInfo(DEFAULT_INSTANCE, "\u0001\u0001����\u0001\u0001\u0001��\u0001��\u0001\u001b", objects);
                case GET_DEFAULT_INSTANCE:
                    return DEFAULT_INSTANCE;
                case GET_PARSER:
                    Parser<MixtureLinearSimilarity> parser = PARSER;
                    if (parser == null) {
                        synchronized (MixtureLinearSimilarity.class) {
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
            MixtureLinearSimilarity defaultInstance = new MixtureLinearSimilarity();
            DEFAULT_INSTANCE = defaultInstance;
            GeneratedMessageLite.registerDefaultInstance(MixtureLinearSimilarity.class, defaultInstance);
        }

        public static MixtureLinearSimilarity getDefaultInstance() {
            return DEFAULT_INSTANCE;
        }

        public static Parser<MixtureLinearSimilarity> parser() {
            return DEFAULT_INSTANCE.getParserForType();
        }
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/MotionModelsProto$MixtureAffine.class */
    public static final class MixtureAffine extends GeneratedMessageLite<MixtureAffine, Builder> implements MixtureAffineOrBuilder {
        public static final int MODEL_FIELD_NUMBER = 1;
        private Internal.ProtobufList<AffineModel> model_ = emptyProtobufList();
        private static final MixtureAffine DEFAULT_INSTANCE;
        private static volatile Parser<MixtureAffine> PARSER;

        private MixtureAffine() {
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.MixtureAffineOrBuilder
        public List<AffineModel> getModelList() {
            return this.model_;
        }

        public List<? extends AffineModelOrBuilder> getModelOrBuilderList() {
            return this.model_;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.MixtureAffineOrBuilder
        public int getModelCount() {
            return this.model_.size();
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.MixtureAffineOrBuilder
        public AffineModel getModel(int index) {
            return this.model_.get(index);
        }

        public AffineModelOrBuilder getModelOrBuilder(int index) {
            return this.model_.get(index);
        }

        private void ensureModelIsMutable() {
            if (!this.model_.isModifiable()) {
                this.model_ = GeneratedMessageLite.mutableCopy(this.model_);
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setModel(int index, AffineModel value) {
            value.getClass();
            ensureModelIsMutable();
            this.model_.set(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addModel(AffineModel value) {
            value.getClass();
            ensureModelIsMutable();
            this.model_.add(value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addModel(int index, AffineModel value) {
            value.getClass();
            ensureModelIsMutable();
            this.model_.add(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addAllModel(Iterable<? extends AffineModel> values) {
            ensureModelIsMutable();
            AbstractMessageLite.addAll((Iterable) values, (List) this.model_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearModel() {
            this.model_ = emptyProtobufList();
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void removeModel(int index) {
            ensureModelIsMutable();
            this.model_.remove(index);
        }

        public static MixtureAffine parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
            return (MixtureAffine) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static MixtureAffine parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (MixtureAffine) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static MixtureAffine parseFrom(ByteString data) throws InvalidProtocolBufferException {
            return (MixtureAffine) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static MixtureAffine parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (MixtureAffine) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static MixtureAffine parseFrom(byte[] data) throws InvalidProtocolBufferException {
            return (MixtureAffine) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static MixtureAffine parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (MixtureAffine) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static MixtureAffine parseFrom(InputStream input) throws IOException {
            return (MixtureAffine) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static MixtureAffine parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (MixtureAffine) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static MixtureAffine parseDelimitedFrom(InputStream input) throws IOException {
            return (MixtureAffine) parseDelimitedFrom(DEFAULT_INSTANCE, input);
        }

        public static MixtureAffine parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (MixtureAffine) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static MixtureAffine parseFrom(CodedInputStream input) throws IOException {
            return (MixtureAffine) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static MixtureAffine parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (MixtureAffine) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Builder newBuilder() {
            return DEFAULT_INSTANCE.createBuilder();
        }

        public static Builder newBuilder(MixtureAffine prototype) {
            return DEFAULT_INSTANCE.createBuilder(prototype);
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/MotionModelsProto$MixtureAffine$Builder.class */
        public static final class Builder extends GeneratedMessageLite.Builder<MixtureAffine, Builder> implements MixtureAffineOrBuilder {
            private Builder() {
                super(MixtureAffine.DEFAULT_INSTANCE);
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.MixtureAffineOrBuilder
            public List<AffineModel> getModelList() {
                return Collections.unmodifiableList(((MixtureAffine) this.instance).getModelList());
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.MixtureAffineOrBuilder
            public int getModelCount() {
                return ((MixtureAffine) this.instance).getModelCount();
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.MixtureAffineOrBuilder
            public AffineModel getModel(int index) {
                return ((MixtureAffine) this.instance).getModel(index);
            }

            public Builder setModel(int index, AffineModel value) {
                copyOnWrite();
                ((MixtureAffine) this.instance).setModel(index, value);
                return this;
            }

            public Builder setModel(int index, AffineModel.Builder builderForValue) {
                copyOnWrite();
                ((MixtureAffine) this.instance).setModel(index, builderForValue.build());
                return this;
            }

            public Builder addModel(AffineModel value) {
                copyOnWrite();
                ((MixtureAffine) this.instance).addModel(value);
                return this;
            }

            public Builder addModel(int index, AffineModel value) {
                copyOnWrite();
                ((MixtureAffine) this.instance).addModel(index, value);
                return this;
            }

            public Builder addModel(AffineModel.Builder builderForValue) {
                copyOnWrite();
                ((MixtureAffine) this.instance).addModel(builderForValue.build());
                return this;
            }

            public Builder addModel(int index, AffineModel.Builder builderForValue) {
                copyOnWrite();
                ((MixtureAffine) this.instance).addModel(index, builderForValue.build());
                return this;
            }

            public Builder addAllModel(Iterable<? extends AffineModel> values) {
                copyOnWrite();
                ((MixtureAffine) this.instance).addAllModel(values);
                return this;
            }

            public Builder clearModel() {
                copyOnWrite();
                ((MixtureAffine) this.instance).clearModel();
                return this;
            }

            public Builder removeModel(int index) {
                copyOnWrite();
                ((MixtureAffine) this.instance).removeModel(index);
                return this;
            }
        }

        /* JADX INFO: Access modifiers changed from: protected */
        @Override // com.google.protobuf.GeneratedMessageLite
        public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
            switch (method) {
                case NEW_MUTABLE_INSTANCE:
                    return new MixtureAffine();
                case NEW_BUILDER:
                    return new Builder();
                case BUILD_MESSAGE_INFO:
                    Object[] objects = {"model_", AffineModel.class};
                    return newMessageInfo(DEFAULT_INSTANCE, "\u0001\u0001����\u0001\u0001\u0001��\u0001��\u0001\u001b", objects);
                case GET_DEFAULT_INSTANCE:
                    return DEFAULT_INSTANCE;
                case GET_PARSER:
                    Parser<MixtureAffine> parser = PARSER;
                    if (parser == null) {
                        synchronized (MixtureAffine.class) {
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
            MixtureAffine defaultInstance = new MixtureAffine();
            DEFAULT_INSTANCE = defaultInstance;
            GeneratedMessageLite.registerDefaultInstance(MixtureAffine.class, defaultInstance);
        }

        public static MixtureAffine getDefaultInstance() {
            return DEFAULT_INSTANCE;
        }

        public static Parser<MixtureAffine> parser() {
            return DEFAULT_INSTANCE.getParserForType();
        }
    }

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/MotionModelsProto$MixtureHomography.class */
    public static final class MixtureHomography extends GeneratedMessageLite<MixtureHomography, Builder> implements MixtureHomographyOrBuilder {
        private int bitField0_;
        public static final int MODEL_FIELD_NUMBER = 1;
        private Internal.ProtobufList<Homography> model_ = emptyProtobufList();
        public static final int DOF_FIELD_NUMBER = 2;
        private int dof_;
        private static final MixtureHomography DEFAULT_INSTANCE;
        private static volatile Parser<MixtureHomography> PARSER;

        private MixtureHomography() {
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/MotionModelsProto$MixtureHomography$VariableDOF.class */
        public enum VariableDOF implements Internal.EnumLite {
            ALL_DOF(0),
            TRANSLATION_DOF(1),
            SKEW_ROTATION_DOF(2),
            CONST_DOF(3);
            
            public static final int ALL_DOF_VALUE = 0;
            public static final int TRANSLATION_DOF_VALUE = 1;
            public static final int SKEW_ROTATION_DOF_VALUE = 2;
            public static final int CONST_DOF_VALUE = 3;
            private static final Internal.EnumLiteMap<VariableDOF> internalValueMap = new Internal.EnumLiteMap<VariableDOF>() { // from class: com.google.mediapipe.tracking.MotionModelsProto.MixtureHomography.VariableDOF.1
                /* JADX DEBUG: Method merged with bridge method */
                /* JADX WARN: Can't rename method to resolve collision */
                @Override // com.google.protobuf.Internal.EnumLiteMap
                public VariableDOF findValueByNumber(int number) {
                    return VariableDOF.forNumber(number);
                }
            };
            private final int value;

            @Override // com.google.protobuf.Internal.EnumLite
            public final int getNumber() {
                return this.value;
            }

            @Deprecated
            public static VariableDOF valueOf(int value) {
                return forNumber(value);
            }

            public static VariableDOF forNumber(int value) {
                switch (value) {
                    case 0:
                        return ALL_DOF;
                    case 1:
                        return TRANSLATION_DOF;
                    case 2:
                        return SKEW_ROTATION_DOF;
                    case 3:
                        return CONST_DOF;
                    default:
                        return null;
                }
            }

            public static Internal.EnumLiteMap<VariableDOF> internalGetValueMap() {
                return internalValueMap;
            }

            public static Internal.EnumVerifier internalGetVerifier() {
                return VariableDOFVerifier.INSTANCE;
            }

            /* JADX INFO: Access modifiers changed from: private */
            /* JADX WARN: Classes with same name are omitted:
              classes2.dex
             */
            /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/MotionModelsProto$MixtureHomography$VariableDOF$VariableDOFVerifier.class */
            public static final class VariableDOFVerifier implements Internal.EnumVerifier {
                static final Internal.EnumVerifier INSTANCE = new VariableDOFVerifier();

                private VariableDOFVerifier() {
                }

                @Override // com.google.protobuf.Internal.EnumVerifier
                public boolean isInRange(int number) {
                    return VariableDOF.forNumber(number) != null;
                }
            }

            VariableDOF(int value) {
                this.value = value;
            }
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.MixtureHomographyOrBuilder
        public List<Homography> getModelList() {
            return this.model_;
        }

        public List<? extends HomographyOrBuilder> getModelOrBuilderList() {
            return this.model_;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.MixtureHomographyOrBuilder
        public int getModelCount() {
            return this.model_.size();
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.MixtureHomographyOrBuilder
        public Homography getModel(int index) {
            return this.model_.get(index);
        }

        public HomographyOrBuilder getModelOrBuilder(int index) {
            return this.model_.get(index);
        }

        private void ensureModelIsMutable() {
            if (!this.model_.isModifiable()) {
                this.model_ = GeneratedMessageLite.mutableCopy(this.model_);
            }
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setModel(int index, Homography value) {
            value.getClass();
            ensureModelIsMutable();
            this.model_.set(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addModel(Homography value) {
            value.getClass();
            ensureModelIsMutable();
            this.model_.add(value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addModel(int index, Homography value) {
            value.getClass();
            ensureModelIsMutable();
            this.model_.add(index, value);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void addAllModel(Iterable<? extends Homography> values) {
            ensureModelIsMutable();
            AbstractMessageLite.addAll((Iterable) values, (List) this.model_);
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearModel() {
            this.model_ = emptyProtobufList();
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void removeModel(int index) {
            ensureModelIsMutable();
            this.model_.remove(index);
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.MixtureHomographyOrBuilder
        public boolean hasDof() {
            return (this.bitField0_ & 1) != 0;
        }

        @Override // com.google.mediapipe.tracking.MotionModelsProto.MixtureHomographyOrBuilder
        public VariableDOF getDof() {
            VariableDOF result = VariableDOF.forNumber(this.dof_);
            return result == null ? VariableDOF.ALL_DOF : result;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void setDof(VariableDOF value) {
            this.dof_ = value.getNumber();
            this.bitField0_ |= 1;
        }

        /* JADX INFO: Access modifiers changed from: private */
        public void clearDof() {
            this.bitField0_ &= -2;
            this.dof_ = 0;
        }

        public static MixtureHomography parseFrom(ByteBuffer data) throws InvalidProtocolBufferException {
            return (MixtureHomography) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static MixtureHomography parseFrom(ByteBuffer data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (MixtureHomography) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static MixtureHomography parseFrom(ByteString data) throws InvalidProtocolBufferException {
            return (MixtureHomography) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static MixtureHomography parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (MixtureHomography) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static MixtureHomography parseFrom(byte[] data) throws InvalidProtocolBufferException {
            return (MixtureHomography) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data);
        }

        public static MixtureHomography parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return (MixtureHomography) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, data, extensionRegistry);
        }

        public static MixtureHomography parseFrom(InputStream input) throws IOException {
            return (MixtureHomography) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static MixtureHomography parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (MixtureHomography) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static MixtureHomography parseDelimitedFrom(InputStream input) throws IOException {
            return (MixtureHomography) parseDelimitedFrom(DEFAULT_INSTANCE, input);
        }

        public static MixtureHomography parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (MixtureHomography) parseDelimitedFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static MixtureHomography parseFrom(CodedInputStream input) throws IOException {
            return (MixtureHomography) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input);
        }

        public static MixtureHomography parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return (MixtureHomography) GeneratedMessageLite.parseFrom(DEFAULT_INSTANCE, input, extensionRegistry);
        }

        public static Builder newBuilder() {
            return DEFAULT_INSTANCE.createBuilder();
        }

        public static Builder newBuilder(MixtureHomography prototype) {
            return DEFAULT_INSTANCE.createBuilder(prototype);
        }

        /* JADX WARN: Classes with same name are omitted:
          classes2.dex
         */
        /* loaded from: base.apk:classes.jar:com/google/mediapipe/tracking/MotionModelsProto$MixtureHomography$Builder.class */
        public static final class Builder extends GeneratedMessageLite.Builder<MixtureHomography, Builder> implements MixtureHomographyOrBuilder {
            private Builder() {
                super(MixtureHomography.DEFAULT_INSTANCE);
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.MixtureHomographyOrBuilder
            public List<Homography> getModelList() {
                return Collections.unmodifiableList(((MixtureHomography) this.instance).getModelList());
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.MixtureHomographyOrBuilder
            public int getModelCount() {
                return ((MixtureHomography) this.instance).getModelCount();
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.MixtureHomographyOrBuilder
            public Homography getModel(int index) {
                return ((MixtureHomography) this.instance).getModel(index);
            }

            public Builder setModel(int index, Homography value) {
                copyOnWrite();
                ((MixtureHomography) this.instance).setModel(index, value);
                return this;
            }

            public Builder setModel(int index, Homography.Builder builderForValue) {
                copyOnWrite();
                ((MixtureHomography) this.instance).setModel(index, builderForValue.build());
                return this;
            }

            public Builder addModel(Homography value) {
                copyOnWrite();
                ((MixtureHomography) this.instance).addModel(value);
                return this;
            }

            public Builder addModel(int index, Homography value) {
                copyOnWrite();
                ((MixtureHomography) this.instance).addModel(index, value);
                return this;
            }

            public Builder addModel(Homography.Builder builderForValue) {
                copyOnWrite();
                ((MixtureHomography) this.instance).addModel(builderForValue.build());
                return this;
            }

            public Builder addModel(int index, Homography.Builder builderForValue) {
                copyOnWrite();
                ((MixtureHomography) this.instance).addModel(index, builderForValue.build());
                return this;
            }

            public Builder addAllModel(Iterable<? extends Homography> values) {
                copyOnWrite();
                ((MixtureHomography) this.instance).addAllModel(values);
                return this;
            }

            public Builder clearModel() {
                copyOnWrite();
                ((MixtureHomography) this.instance).clearModel();
                return this;
            }

            public Builder removeModel(int index) {
                copyOnWrite();
                ((MixtureHomography) this.instance).removeModel(index);
                return this;
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.MixtureHomographyOrBuilder
            public boolean hasDof() {
                return ((MixtureHomography) this.instance).hasDof();
            }

            @Override // com.google.mediapipe.tracking.MotionModelsProto.MixtureHomographyOrBuilder
            public VariableDOF getDof() {
                return ((MixtureHomography) this.instance).getDof();
            }

            public Builder setDof(VariableDOF value) {
                copyOnWrite();
                ((MixtureHomography) this.instance).setDof(value);
                return this;
            }

            public Builder clearDof() {
                copyOnWrite();
                ((MixtureHomography) this.instance).clearDof();
                return this;
            }
        }

        /* JADX INFO: Access modifiers changed from: protected */
        @Override // com.google.protobuf.GeneratedMessageLite
        public final Object dynamicMethod(GeneratedMessageLite.MethodToInvoke method, Object arg0, Object arg1) {
            switch (method) {
                case NEW_MUTABLE_INSTANCE:
                    return new MixtureHomography();
                case NEW_BUILDER:
                    return new Builder();
                case BUILD_MESSAGE_INFO:
                    Object[] objects = {"bitField0_", "model_", Homography.class, "dof_", VariableDOF.internalGetVerifier()};
                    return newMessageInfo(DEFAULT_INSTANCE, "\u0001\u0002��\u0001\u0001\u0002\u0002��\u0001��\u0001\u001b\u0002\f��", objects);
                case GET_DEFAULT_INSTANCE:
                    return DEFAULT_INSTANCE;
                case GET_PARSER:
                    Parser<MixtureHomography> parser = PARSER;
                    if (parser == null) {
                        synchronized (MixtureHomography.class) {
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
            MixtureHomography defaultInstance = new MixtureHomography();
            DEFAULT_INSTANCE = defaultInstance;
            GeneratedMessageLite.registerDefaultInstance(MixtureHomography.class, defaultInstance);
        }

        public static MixtureHomography getDefaultInstance() {
            return DEFAULT_INSTANCE;
        }

        public static Parser<MixtureHomography> parser() {
            return DEFAULT_INSTANCE.getParserForType();
        }
    }
}