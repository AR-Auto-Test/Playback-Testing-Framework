package com.google.mediapipe.framework;

import com.google.protobuf.ExtensionRegistryLite;
import com.google.protobuf.Internal;
import com.google.protobuf.InvalidProtocolBufferException;
import com.google.protobuf.MessageLite;
import java.util.NoSuchElementException;

/* JADX WARN: Classes with same name are omitted:
  classes2.dex
 */
/* loaded from: base.apk:classes.jar:com/google/mediapipe/framework/ProtoUtil.class */
public final class ProtoUtil {
    static TypeNameRegistry typeNameRegistry = new TypeNameRegistryConcrete();

    /* JADX WARN: Classes with same name are omitted:
      classes2.dex
     */
    /* loaded from: base.apk:classes.jar:com/google/mediapipe/framework/ProtoUtil$SerializedMessage.class */
    static class SerializedMessage {
        public String typeName;
        public byte[] value;
    }

    public static <T extends MessageLite> void registerTypeName(Class<T> clazz, String typeName) {
        typeNameRegistry.registerTypeName(clazz, typeName);
    }

    public static <T extends MessageLite> String getTypeName(Class<T> clazz) {
        return typeNameRegistry.getTypeName(clazz);
    }

    public static ExtensionRegistryLite getExtensionRegistry() {
        return ExtensionRegistryLite.getEmptyRegistry();
    }

    public static <T extends MessageLite> SerializedMessage pack(T message) {
        SerializedMessage result = new SerializedMessage();
        result.typeName = getTypeName(message.getClass());
        if (result.typeName == null) {
            throw new NoSuchElementException("Cannot determine the protobuf type name for class: " + message.getClass() + ". Have you called ProtoUtil.registerTypeName?");
        }
        result.value = message.toByteArray();
        return result;
    }

    public static <T extends MessageLite> T unpack(SerializedMessage serialized, Class<T> clazz) throws InvalidProtocolBufferException {
        MessageLite defaultInstance = Internal.getDefaultInstance(clazz);
        String expectedType = getTypeName(defaultInstance.getClass());
        if (!serialized.typeName.equals(expectedType)) {
            throw new InvalidProtocolBufferException("Message type does not match the expected type. Expected: " + expectedType + " Got: " + serialized.typeName);
        }
        T result = (T) defaultInstance.getParserForType().parseFrom(serialized.value, getExtensionRegistry());
        return result;
    }

    private ProtoUtil() {
    }
}