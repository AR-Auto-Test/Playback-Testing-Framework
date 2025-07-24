package b.d.b.d1;

import b.d.b.d1.b1;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/* compiled from: UseCaseAttachState.java */
/* loaded from: classes.dex */
public final class h1 {

    /* renamed from: a  reason: collision with root package name */
    public final String f1485a;

    /* renamed from: b  reason: collision with root package name */
    public final Map<String, b> f1486b = new HashMap();

    /* compiled from: UseCaseAttachState.java */
    /* loaded from: classes.dex */
    public interface a {
        boolean a(b bVar);
    }

    /* compiled from: UseCaseAttachState.java */
    /* loaded from: classes.dex */
    public static final class b {

        /* renamed from: a  reason: collision with root package name */
        public final b1 f1487a;

        /* renamed from: b  reason: collision with root package name */
        public boolean f1488b = false;

        /* renamed from: c  reason: collision with root package name */
        public boolean f1489c = false;

        public b(b1 b1Var) {
            this.f1487a = b1Var;
        }
    }

    public h1(String str) {
        this.f1485a = str;
    }

    public b1.f a() {
        b1.f fVar = new b1.f();
        ArrayList arrayList = new ArrayList();
        for (Map.Entry<String, b> entry : this.f1486b.entrySet()) {
            b value = entry.getValue();
            if (value.f1488b) {
                fVar.a(value.f1487a);
                arrayList.add(entry.getKey());
            }
        }
        b.d.b.u0.a("UseCaseAttachState", "All use case: " + arrayList + " for camera: " + this.f1485a, null);
        return fVar;
    }

    public Collection<b1> b() {
        return Collections.unmodifiableCollection(c(j.f1496a));
    }

    public final Collection<b1> c(a aVar) {
        ArrayList arrayList = new ArrayList();
        for (Map.Entry<String, b> entry : this.f1486b.entrySet()) {
            if (aVar.a(entry.getValue())) {
                arrayList.add(entry.getValue().f1487a);
            }
        }
        return arrayList;
    }

    public boolean d(String str) {
        if (this.f1486b.containsKey(str)) {
            return this.f1486b.get(str).f1488b;
        }
        return false;
    }

    public void e(String str, b1 b1Var) {
        b bVar = this.f1486b.get(str);
        if (bVar == null) {
            bVar = new b(b1Var);
            this.f1486b.put(str, bVar);
        }
        bVar.f1489c = true;
    }

    public void f(String str, b1 b1Var) {
        b bVar = this.f1486b.get(str);
        if (bVar == null) {
            bVar = new b(b1Var);
            this.f1486b.put(str, bVar);
        }
        bVar.f1488b = true;
    }

    public void g(String str) {
        if (this.f1486b.containsKey(str)) {
            b bVar = this.f1486b.get(str);
            bVar.f1489c = false;
            if (bVar.f1488b) {
                return;
            }
            this.f1486b.remove(str);
        }
    }

    public void h(String str, b1 b1Var) {
        if (this.f1486b.containsKey(str)) {
            b bVar = new b(b1Var);
            b bVar2 = this.f1486b.get(str);
            bVar.f1488b = bVar2.f1488b;
            bVar.f1489c = bVar2.f1489c;
            this.f1486b.put(str, bVar);
        }
    }
}