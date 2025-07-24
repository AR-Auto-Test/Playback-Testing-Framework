package b.d.b.d1;

/* compiled from: SingleImageProxyBundle.java */
/* loaded from: classes.dex */
public final class c1 {

    /* renamed from: a  reason: collision with root package name */
    public final b.d.b.r0 f1441a;

    public c1(b.d.b.r0 r0Var, String str) {
        b.d.b.q0 n = r0Var.n();
        if (n != null) {
            Integer num = n.a().f1480b.get(str);
            if (num != null) {
                num.intValue();
                this.f1441a = r0Var;
                return;
            }
            throw new IllegalArgumentException("ImageProxy has no associated tag");
        }
        throw new IllegalArgumentException("ImageProxy has no associated ImageInfo");
    }
}