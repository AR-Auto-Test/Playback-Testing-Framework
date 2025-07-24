package b.j.j;

import android.view.View;
import b.j.j.q;

/* compiled from: ViewCompat.java */
/* loaded from: classes.dex */
public class p extends q.a<Boolean> {
    public p(int i, Class cls, int i2) {
        super(i, cls, i2);
    }

    /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
    @Override // b.j.j.q.a
    public Boolean b(View view) {
        return Boolean.valueOf(view.isAccessibilityHeading());
    }

    public void d(View view, Object obj) {
        view.setAccessibilityHeading(((Boolean) obj).booleanValue());
    }

    public boolean e(Object obj, Object obj2) {
        return !a((Boolean) obj, (Boolean) obj2);
    }
}