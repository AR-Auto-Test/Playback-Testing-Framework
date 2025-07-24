package c.a.a.z;

import c.a.a.z.k.m;
import java.util.List;

/* compiled from: FontCharacter.java */
/* loaded from: classes.dex */
public class d {

    /* renamed from: a  reason: collision with root package name */
    public final List<m> f3272a;

    /* renamed from: b  reason: collision with root package name */
    public final char f3273b;

    /* renamed from: c  reason: collision with root package name */
    public final double f3274c;

    /* renamed from: d  reason: collision with root package name */
    public final String f3275d;

    /* renamed from: e  reason: collision with root package name */
    public final String f3276e;

    public d(List<m> list, char c2, double d2, double d3, String str, String str2) {
        this.f3272a = list;
        this.f3273b = c2;
        this.f3274c = d3;
        this.f3275d = str;
        this.f3276e = str2;
    }

    public static int a(char c2, String str, String str2) {
        int hashCode = str.hashCode();
        return str2.hashCode() + ((hashCode + ((0 + c2) * 31)) * 31);
    }

    public int hashCode() {
        return a(this.f3273b, this.f3276e, this.f3275d);
    }
}