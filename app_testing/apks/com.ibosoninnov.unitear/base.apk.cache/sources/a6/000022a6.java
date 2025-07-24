package com.google.common.flogger;

import com.google.common.flogger.backend.KeyValueHandler;
import com.google.common.flogger.util.Checks;

/* loaded from: classes.dex */
public class MetadataKey<T> {
    private final boolean canRepeat;
    private final Class<T> clazz;
    private final String label;

    public MetadataKey(String str, Class<T> cls, boolean z) {
        this.label = Checks.checkMetadataIdentifier(str);
        this.clazz = (Class) Checks.checkNotNull(cls, "class");
        this.canRepeat = z;
    }

    public static <T> MetadataKey<T> repeated(String str, Class<T> cls) {
        return new MetadataKey<>(str, cls, true);
    }

    public static <T> MetadataKey<T> single(String str, Class<T> cls) {
        return new MetadataKey<>(str, cls, false);
    }

    public final boolean canRepeat() {
        return this.canRepeat;
    }

    public final T cast(Object obj) {
        return this.clazz.cast(obj);
    }

    public void emit(Object obj, KeyValueHandler keyValueHandler) {
        keyValueHandler.handle(getLabel(), obj);
    }

    public final boolean equals(Object obj) {
        return super.equals(obj);
    }

    public final String getLabel() {
        return this.label;
    }

    public final int hashCode() {
        return super.hashCode();
    }

    public final String toString() {
        return getClass().getName() + "/" + this.label + "[" + this.clazz.getName() + "]";
    }
}