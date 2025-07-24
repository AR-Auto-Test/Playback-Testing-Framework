package c.c.a;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/* compiled from: GlideExperiments.java */
/* loaded from: classes.dex */
public class e {

    /* renamed from: a  reason: collision with root package name */
    public final Map<Class<?>, ?> f3433a;

    /* compiled from: GlideExperiments.java */
    /* loaded from: classes.dex */
    public static final class a {

        /* renamed from: a  reason: collision with root package name */
        public final Map<Class<?>, ?> f3434a = new HashMap();
    }

    public e(a aVar) {
        this.f3433a = Collections.unmodifiableMap(new HashMap(aVar.f3434a));
    }
}