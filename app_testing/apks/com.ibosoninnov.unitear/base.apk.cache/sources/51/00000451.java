package b.j.f;

import android.os.LocaleList;
import java.util.Locale;

/* compiled from: LocaleListCompat.java */
/* loaded from: classes.dex */
public final class c {

    /* renamed from: a  reason: collision with root package name */
    public d f2120a;

    static {
        new LocaleList(new Locale[0]);
    }

    public boolean equals(Object obj) {
        return (obj instanceof c) && this.f2120a.equals(((c) obj).f2120a);
    }

    public int hashCode() {
        return this.f2120a.hashCode();
    }

    public String toString() {
        return this.f2120a.toString();
    }
}