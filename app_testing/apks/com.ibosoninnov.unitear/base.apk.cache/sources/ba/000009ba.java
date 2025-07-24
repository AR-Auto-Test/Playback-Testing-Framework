package c.e.b;

import f.v;
import f.y;
import java.util.Objects;
import java.util.concurrent.TimeUnit;

/* compiled from: HttpHelper.java */
/* loaded from: classes2.dex */
public class cc {

    /* renamed from: a  reason: collision with root package name */
    public static f.v f4613a;

    /* renamed from: b  reason: collision with root package name */
    public cc f4614b;

    /* renamed from: c  reason: collision with root package name */
    public String f4615c = "";

    /* renamed from: d  reason: collision with root package name */
    public a f4616d;

    /* compiled from: HttpHelper.java */
    /* loaded from: classes2.dex */
    public interface a {
        void a(String str);

        void b(String str);
    }

    public cc(a aVar) {
        this.f4616d = aVar;
    }

    public void a(String str, a aVar) {
        f.y a2;
        if (this.f4614b == null) {
            this.f4614b = new cc(aVar);
        }
        cc ccVar = this.f4614b;
        Objects.requireNonNull(ccVar);
        f.v vVar = f4613a;
        if (vVar != null) {
            vVar.f6122d.a();
        }
        v.b bVar = new v.b();
        TimeUnit timeUnit = TimeUnit.SECONDS;
        bVar.a(10L, timeUnit);
        bVar.c(10L, timeUnit);
        bVar.b(15L, timeUnit);
        f4613a = new f.v(bVar);
        if (ac.f4547a.f4552f) {
            y.a aVar2 = new y.a();
            aVar2.d(str);
            a2 = aVar2.a();
        } else {
            y.a aVar3 = new y.a();
            aVar3.d(str);
            a2 = aVar3.a();
        }
        ((f.x) f4613a.a(a2)).b(new bc(ccVar));
    }

    public cc() {
    }
}