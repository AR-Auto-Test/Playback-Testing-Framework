package b.b.c;

import android.app.Dialog;
import android.os.Bundle;

/* compiled from: AppCompatDialogFragment.java */
/* loaded from: classes.dex */
public class q extends b.q.b.c {
    @Override // b.q.b.c
    public Dialog onCreateDialog(Bundle bundle) {
        return new p(getContext(), getTheme());
    }

    @Override // b.q.b.c
    public void setupDialog(Dialog dialog, int i) {
        if (dialog instanceof p) {
            p pVar = (p) dialog;
            if (i != 1 && i != 2) {
                if (i != 3) {
                    return;
                }
                dialog.getWindow().addFlags(24);
            }
            pVar.supportRequestWindowFeature(1);
            return;
        }
        super.setupDialog(dialog, i);
    }
}