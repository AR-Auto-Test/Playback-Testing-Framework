package com.google.common.flogger;

import com.google.common.flogger.util.Checks;

/* loaded from: classes.dex */
public final class LazyArgs {
    private LazyArgs() {
    }

    public static <T> LazyArg<T> lazy(LazyArg<T> lazyArg) {
        return (LazyArg) Checks.checkNotNull(lazyArg, "lazy arg");
    }
}