package com.google.mediapipe.framework;

import com.google.protobuf.MessageLite;

/* JADX WARN: Classes with same name are omitted:
  classes2.dex
 */
/* loaded from: base.apk:classes.jar:com/google/mediapipe/framework/TypeNameRegistry.class */
interface TypeNameRegistry {
    <T extends MessageLite> String getTypeName(Class<T> clazz);

    <T extends MessageLite> void registerTypeName(Class<T> clazz, String typeName);
}