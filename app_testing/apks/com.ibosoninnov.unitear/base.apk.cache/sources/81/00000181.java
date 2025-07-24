package b.b.g.i;

import android.content.DialogInterface;
import android.view.KeyEvent;
import android.view.View;
import android.view.Window;
import b.b.g.i.e;
import b.b.g.i.m;

/* compiled from: MenuDialogHelper.java */
/* loaded from: classes.dex */
public class h implements DialogInterface.OnKeyListener, DialogInterface.OnClickListener, DialogInterface.OnDismissListener, m.a {

    /* renamed from: b  reason: collision with root package name */
    public g f727b;

    /* renamed from: c  reason: collision with root package name */
    public b.b.c.g f728c;

    /* renamed from: d  reason: collision with root package name */
    public e f729d;

    public h(g gVar) {
        this.f727b = gVar;
    }

    @Override // b.b.g.i.m.a
    public boolean a(g gVar) {
        return false;
    }

    @Override // android.content.DialogInterface.OnClickListener
    public void onClick(DialogInterface dialogInterface, int i) {
        this.f727b.performItemAction((i) ((e.a) this.f729d.a()).getItem(i), 0);
    }

    @Override // b.b.g.i.m.a
    public void onCloseMenu(g gVar, boolean z) {
        b.b.c.g gVar2;
        if ((z || gVar == this.f727b) && (gVar2 = this.f728c) != null) {
            gVar2.dismiss();
        }
    }

    @Override // android.content.DialogInterface.OnDismissListener
    public void onDismiss(DialogInterface dialogInterface) {
        e eVar = this.f729d;
        g gVar = this.f727b;
        m.a aVar = eVar.f717f;
        if (aVar != null) {
            aVar.onCloseMenu(gVar, true);
        }
    }

    @Override // android.content.DialogInterface.OnKeyListener
    public boolean onKey(DialogInterface dialogInterface, int i, KeyEvent keyEvent) {
        Window window;
        View decorView;
        KeyEvent.DispatcherState keyDispatcherState;
        View decorView2;
        KeyEvent.DispatcherState keyDispatcherState2;
        if (i == 82 || i == 4) {
            if (keyEvent.getAction() == 0 && keyEvent.getRepeatCount() == 0) {
                Window window2 = this.f728c.getWindow();
                if (window2 != null && (decorView2 = window2.getDecorView()) != null && (keyDispatcherState2 = decorView2.getKeyDispatcherState()) != null) {
                    keyDispatcherState2.startTracking(keyEvent, this);
                    return true;
                }
            } else if (keyEvent.getAction() == 1 && !keyEvent.isCanceled() && (window = this.f728c.getWindow()) != null && (decorView = window.getDecorView()) != null && (keyDispatcherState = decorView.getKeyDispatcherState()) != null && keyDispatcherState.isTracking(keyEvent)) {
                this.f727b.close(true);
                dialogInterface.dismiss();
                return true;
            }
        }
        return this.f727b.performShortcut(i, keyEvent, 0);
    }
}