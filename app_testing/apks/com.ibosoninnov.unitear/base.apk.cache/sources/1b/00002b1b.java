package com.google.protobuf;

import c.b.a.a.a;

/* loaded from: classes2.dex */
public class GeneratedMessageInfoFactory implements MessageInfoFactory {
    private static final GeneratedMessageInfoFactory instance = new GeneratedMessageInfoFactory();

    private GeneratedMessageInfoFactory() {
    }

    public static GeneratedMessageInfoFactory getInstance() {
        return instance;
    }

    @Override // com.google.protobuf.MessageInfoFactory
    public boolean isSupported(Class<?> cls) {
        return GeneratedMessageLite.class.isAssignableFrom(cls);
    }

    @Override // com.google.protobuf.MessageInfoFactory
    public MessageInfo messageInfoFor(Class<?> cls) {
        if (GeneratedMessageLite.class.isAssignableFrom(cls)) {
            try {
                return (MessageInfo) GeneratedMessageLite.getDefaultInstance(cls.asSubclass(GeneratedMessageLite.class)).buildMessageInfo();
            } catch (Exception e2) {
                StringBuilder x = a.x("Unable to get message info for ");
                x.append(cls.getName());
                throw new RuntimeException(x.toString(), e2);
            }
        }
        StringBuilder x2 = a.x("Unsupported message type: ");
        x2.append(cls.getName());
        throw new IllegalArgumentException(x2.toString());
    }
}