package com.google.mediapipe.framework;

import com.google.mediapipe.framework.MediaPipeException;
import com.google.protobuf.MessageLite;
import java.util.HashMap;
import java.util.Map;

/* JADX WARN: Classes with same name are omitted:
  classes2.dex
 */
/* compiled from: TypeNameRegistryLite.java */
/* loaded from: base.apk:classes.jar:com/google/mediapipe/framework/TypeNameRegistryConcrete.class */
final class TypeNameRegistryConcrete implements TypeNameRegistry {
    private final Map<Class<? extends MessageLite>, String> typeNames = new HashMap();

    @Override // com.google.mediapipe.framework.TypeNameRegistry
    public <T extends MessageLite> String getTypeName(Class<T> javaClass) {
        return this.typeNames.get(javaClass);
    }

    @Override // com.google.mediapipe.framework.TypeNameRegistry
    public <T extends MessageLite> void registerTypeName(Class<T> clazz, String typeName) {
        if (this.typeNames.containsKey(clazz) && !this.typeNames.get(clazz).equals(typeName)) {
            throw new MediaPipeException(MediaPipeException.StatusCode.ALREADY_EXISTS.ordinal(), "Protobuf type name: " + typeName + " conflicts with: " + this.typeNames.get(clazz));
        }
        this.typeNames.put(clazz, typeName);
    }
}