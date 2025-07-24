package b.j.j;

import android.view.View;
import b.j.j.q;

/* compiled from: ViewCompat.java */
/* loaded from: classes.dex */
public class o extends q.a<CharSequence> {
    public o(int i, Class cls, int i2, int i3) {
        super(i, cls, i2, i3);
    }

    /* JADX DEBUG: Return type fixed from 'java.lang.Object' to match base method */
    @Override // b.j.j.q.a
    public CharSequence b(View view) {
        return view.getStateDescription();
    }
}