package c.a.a.z;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/* compiled from: KeyPath.java */
/* loaded from: classes.dex */
public class e {

    /* renamed from: a  reason: collision with root package name */
    public final List<String> f3277a;

    /* renamed from: b  reason: collision with root package name */
    public f f3278b;

    public e(String... strArr) {
        this.f3277a = Arrays.asList(strArr);
    }

    public e a(String str) {
        e eVar = new e(this);
        eVar.f3277a.add(str);
        return eVar;
    }

    public final boolean b() {
        List<String> list = this.f3277a;
        return list.get(list.size() - 1).equals("**");
    }

    public boolean c(String str, int i) {
        if (i >= this.f3277a.size()) {
            return false;
        }
        boolean z = i == this.f3277a.size() - 1;
        String str2 = this.f3277a.get(i);
        if (!str2.equals("**")) {
            return (z || (i == this.f3277a.size() + (-2) && b())) && (str2.equals(str) || str2.equals("*"));
        }
        if (!z && this.f3277a.get(i + 1).equals(str)) {
            return i == this.f3277a.size() + (-2) || (i == this.f3277a.size() + (-3) && b());
        } else if (z) {
            return true;
        } else {
            int i2 = i + 1;
            if (i2 < this.f3277a.size() - 1) {
                return false;
            }
            return this.f3277a.get(i2).equals(str);
        }
    }

    public int d(String str, int i) {
        if ("__container".equals(str)) {
            return 0;
        }
        if (this.f3277a.get(i).equals("**")) {
            return (i != this.f3277a.size() - 1 && this.f3277a.get(i + 1).equals(str)) ? 2 : 0;
        }
        return 1;
    }

    public boolean e(String str, int i) {
        if ("__container".equals(str)) {
            return true;
        }
        if (i >= this.f3277a.size()) {
            return false;
        }
        return this.f3277a.get(i).equals(str) || this.f3277a.get(i).equals("**") || this.f3277a.get(i).equals("*");
    }

    public boolean f(String str, int i) {
        return "__container".equals(str) || i < this.f3277a.size() - 1 || this.f3277a.get(i).equals("**");
    }

    public e g(f fVar) {
        e eVar = new e(this);
        eVar.f3278b = fVar;
        return eVar;
    }

    public String toString() {
        StringBuilder x = c.b.a.a.a.x("KeyPath{keys=");
        x.append(this.f3277a);
        x.append(",resolved=");
        x.append(this.f3278b != null);
        x.append('}');
        return x.toString();
    }

    public e(e eVar) {
        this.f3277a = new ArrayList(eVar.f3277a);
        this.f3278b = eVar.f3278b;
    }
}