package com.google.common.base;

import com.google.common.annotations.GwtCompatible;
import com.google.errorprone.annotations.CanIgnoreReturnValue;

@GwtCompatible
/* loaded from: classes.dex */
public interface Predicate<T> {
    @CanIgnoreReturnValue
    boolean apply(T t);

    boolean equals(Object obj);
}