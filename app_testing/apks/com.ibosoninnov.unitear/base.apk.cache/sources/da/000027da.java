package com.google.firebase.tracing;

import com.google.firebase.components.Component;
import com.google.firebase.components.ComponentContainer;
import com.google.firebase.components.ComponentFactory;
import com.google.firebase.components.ComponentRegistrar;
import com.google.firebase.components.ComponentRegistrarProcessor;
import com.google.firebase.tracing.ComponentMonitor;
import com.google.firebase.tracing.FirebaseTrace;
import java.util.ArrayList;
import java.util.List;

/* loaded from: classes2.dex */
public class ComponentMonitor implements ComponentRegistrarProcessor {

    /* renamed from: a  reason: collision with root package name */
    public static final /* synthetic */ int f5649a = 0;

    @Override // com.google.firebase.components.ComponentRegistrarProcessor
    public List<Component<?>> processRegistrar(ComponentRegistrar componentRegistrar) {
        ArrayList arrayList = new ArrayList();
        for (final Component<?> component : componentRegistrar.getComponents()) {
            final String name = component.getName();
            if (name != null) {
                component = component.withFactory(new ComponentFactory() { // from class: c.d.c.m.a
                    @Override // com.google.firebase.components.ComponentFactory
                    public final Object create(ComponentContainer componentContainer) {
                        String str = name;
                        Component component2 = component;
                        int i = ComponentMonitor.f5649a;
                        try {
                            FirebaseTrace.pushTrace(str);
                            return component2.getFactory().create(componentContainer);
                        } finally {
                            FirebaseTrace.popTrace();
                        }
                    }
                });
            }
            arrayList.add(component);
        }
        return arrayList;
    }
}