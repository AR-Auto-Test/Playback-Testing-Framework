package com.google.protobuf;

/* loaded from: classes2.dex */
public interface MessageInfoFactory {
    boolean isSupported(Class<?> cls);

    MessageInfo messageInfoFor(Class<?> cls);
}