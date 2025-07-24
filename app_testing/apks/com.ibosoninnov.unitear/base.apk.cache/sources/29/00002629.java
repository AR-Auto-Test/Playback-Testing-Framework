package com.google.firebase.components;

import c.b.a.a.a;
import java.util.Arrays;
import java.util.List;

/* loaded from: classes.dex */
public class DependencyCycleException extends DependencyException {
    private final List<Component<?>> componentsInCycle;

    /* JADX WARN: Illegal instructions before constructor call */
    /*
        Code decompiled incorrectly, please refer to instructions dump.
    */
    public DependencyCycleException(List<Component<?>> list) {
        super(r0.toString());
        StringBuilder x = a.x("Dependency cycle detected: ");
        x.append(Arrays.toString(list.toArray()));
        this.componentsInCycle = list;
    }

    public List<Component<?>> getComponentsInCycle() {
        return this.componentsInCycle;
    }
}