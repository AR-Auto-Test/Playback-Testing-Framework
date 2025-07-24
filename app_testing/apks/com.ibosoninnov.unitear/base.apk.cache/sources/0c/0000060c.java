package b.z;

import android.view.View;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

/* compiled from: TransitionValues.java */
/* loaded from: classes.dex */
public class p {

    /* renamed from: b  reason: collision with root package name */
    public View f2914b;

    /* renamed from: a  reason: collision with root package name */
    public final Map<String, Object> f2913a = new HashMap();

    /* renamed from: c  reason: collision with root package name */
    public final ArrayList<j> f2915c = new ArrayList<>();

    @Deprecated
    public p() {
    }

    public boolean equals(Object obj) {
        if (obj instanceof p) {
            p pVar = (p) obj;
            return this.f2914b == pVar.f2914b && this.f2913a.equals(pVar.f2913a);
        }
        return false;
    }

    public int hashCode() {
        return this.f2913a.hashCode() + (this.f2914b.hashCode() * 31);
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("TransitionValues@");
        x.append(Integer.toHexString(hashCode()));
        x.append(":\n");
        StringBuilder A = c.b.a.a.a.A(x.toString(), "    view = ");
        A.append(this.f2914b);
        A.append("\n");
        String q = c.b.a.a.a.q(A.toString(), "    values:");
        for (String str : this.f2913a.keySet()) {
            q = q + "    " + str + ": " + this.f2913a.get(str) + "\n";
        }
        return q;
    }

    public p(View view) {
        this.f2914b = view;
    }
}