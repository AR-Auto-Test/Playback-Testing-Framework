package c.e.b;

import f.v;
import f.y;
import java.util.Objects;
import java.util.concurrent.TimeUnit;

/* compiled from: HttpHelperPost.java */
/* loaded from: classes2.dex */
public class ec {

    /* renamed from: a  reason: collision with root package name */
    public static f.v f4690a;

    /* renamed from: b  reason: collision with root package name */
    public ec f4691b;

    /* renamed from: c  reason: collision with root package name */
    public String f4692c = "";

    /* renamed from: d  reason: collision with root package name */
    public a f4693d;

    /* compiled from: HttpHelperPost.java */
    /* loaded from: classes2.dex */
    public interface a {
        void a(String str);

        void b(String str);
    }

    public ec(a aVar) {
        this.f4693d = aVar;
    }

    public void a(String str, f.a0 a0Var, a aVar) {
        if (this.f4691b == null) {
            this.f4691b = new ec(aVar);
        }
        ec ecVar = this.f4691b;
        Objects.requireNonNull(ecVar);
        f.v vVar = f4690a;
        if (vVar != null) {
            vVar.f6122d.a();
        }
        if (ac.f4547a.f4552f) {
            v.b bVar = new v.b();
            bVar.f6130d.add(new xb("unitear_dev", "tQYoO1bqdOlQEI4"));
            TimeUnit timeUnit = TimeUnit.SECONDS;
            bVar.a(10L, timeUnit);
            bVar.c(10L, timeUnit);
            bVar.b(15L, timeUnit);
            f4690a = new f.v(bVar);
        } else {
            v.b bVar2 = new v.b();
            TimeUnit timeUnit2 = TimeUnit.SECONDS;
            bVar2.a(10L, timeUnit2);
            bVar2.c(10L, timeUnit2);
            bVar2.b(15L, timeUnit2);
            f4690a = new f.v(bVar2);
        }
        y.a aVar2 = new y.a();
        aVar2.c("POST", a0Var);
        aVar2.d(str);
        ((f.x) f4690a.a(aVar2.a())).b(new dc(ecVar));
    }

    public ec() {
    }
}