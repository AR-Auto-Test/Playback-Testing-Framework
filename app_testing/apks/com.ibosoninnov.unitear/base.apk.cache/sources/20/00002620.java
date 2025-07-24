package com.google.firebase.components;

import c.d.c.g.o;
import java.util.List;

/* loaded from: classes.dex */
public interface ComponentRegistrarProcessor {
    public static final ComponentRegistrarProcessor NOOP = o.f4405a;

    List<Component<?>> processRegistrar(ComponentRegistrar componentRegistrar);
}