package androidx.activity;

import android.os.Bundle;
import android.view.View;
import android.view.Window;
import b.j.b.e;
import b.t.e;
import b.t.f;
import b.t.h;
import b.t.i;
import b.t.p;
import b.t.y;
import b.t.z;
import b.x.c;

/* loaded from: classes.dex */
public class ComponentActivity extends e implements h, z, c, b.a.c {

    /* renamed from: c  reason: collision with root package name */
    public final i f38c;

    /* renamed from: d  reason: collision with root package name */
    public final b.x.b f39d;

    /* renamed from: e  reason: collision with root package name */
    public y f40e;

    /* renamed from: f  reason: collision with root package name */
    public final OnBackPressedDispatcher f41f;

    /* loaded from: classes.dex */
    public class a implements Runnable {
        public a() {
        }

        @Override // java.lang.Runnable
        public void run() {
            ComponentActivity.super.onBackPressed();
        }
    }

    /* loaded from: classes.dex */
    public static final class b {

        /* renamed from: a  reason: collision with root package name */
        public y f45a;
    }

    public ComponentActivity() {
        i iVar = new i(this);
        this.f38c = iVar;
        this.f39d = new b.x.b(this);
        this.f41f = new OnBackPressedDispatcher(new a());
        if (iVar != null) {
            iVar.a(new f() { // from class: androidx.activity.ComponentActivity.2
                @Override // b.t.f
                public void e(h hVar, e.a aVar) {
                    if (aVar == e.a.ON_STOP) {
                        Window window = ComponentActivity.this.getWindow();
                        View peekDecorView = window != null ? window.peekDecorView() : null;
                        if (peekDecorView != null) {
                            peekDecorView.cancelPendingInputEvents();
                        }
                    }
                }
            });
            iVar.a(new f() { // from class: androidx.activity.ComponentActivity.3
                @Override // b.t.f
                public void e(h hVar, e.a aVar) {
                    if (aVar != e.a.ON_DESTROY || ComponentActivity.this.isChangingConfigurations()) {
                        return;
                    }
                    ComponentActivity.this.getViewModelStore().a();
                }
            });
            return;
        }
        throw new IllegalStateException("getLifecycle() returned null in ComponentActivity's constructor. Please make sure you are lazily constructing your Lifecycle in the first call to getLifecycle() rather than relying on field initialization.");
    }

    @Override // b.a.c
    public final OnBackPressedDispatcher b() {
        return this.f41f;
    }

    @Override // b.t.h
    public b.t.e getLifecycle() {
        return this.f38c;
    }

    @Override // b.x.c
    public final b.x.a getSavedStateRegistry() {
        return this.f39d.f2826b;
    }

    @Override // b.t.z
    public y getViewModelStore() {
        if (getApplication() != null) {
            if (this.f40e == null) {
                b bVar = (b) getLastNonConfigurationInstance();
                if (bVar != null) {
                    this.f40e = bVar.f45a;
                }
                if (this.f40e == null) {
                    this.f40e = new y();
                }
            }
            return this.f40e;
        }
        throw new IllegalStateException("Your activity is not yet attached to the Application instance. You can't request ViewModel before onCreate call.");
    }

    @Override // android.app.Activity
    public void onBackPressed() {
        this.f41f.b();
    }

    @Override // b.j.b.e, android.app.Activity
    public void onCreate(Bundle bundle) {
        super.onCreate(bundle);
        this.f39d.a(bundle);
        p.c(this);
    }

    @Override // android.app.Activity
    public final Object onRetainNonConfigurationInstance() {
        b bVar;
        y yVar = this.f40e;
        if (yVar == null && (bVar = (b) getLastNonConfigurationInstance()) != null) {
            yVar = bVar.f45a;
        }
        if (yVar == null) {
            return null;
        }
        b bVar2 = new b();
        bVar2.f45a = yVar;
        return bVar2;
    }

    @Override // b.j.b.e, android.app.Activity
    public void onSaveInstanceState(Bundle bundle) {
        i iVar = this.f38c;
        if (iVar instanceof i) {
            iVar.f(e.b.CREATED);
        }
        super.onSaveInstanceState(bundle);
        this.f39d.b(bundle);
    }
}