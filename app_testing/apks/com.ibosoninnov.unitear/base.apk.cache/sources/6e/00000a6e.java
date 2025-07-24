package c.e.b;

import android.media.MediaPlayer;
import java.util.function.BiConsumer;

/* compiled from: lambda */
/* loaded from: classes2.dex */
public final /* synthetic */ class j9 implements BiConsumer {

    /* renamed from: a  reason: collision with root package name */
    public static final /* synthetic */ j9 f4932a = new j9();

    @Override // java.util.function.BiConsumer
    public final void accept(Object obj, Object obj2) {
        MediaPlayer mediaPlayer = (MediaPlayer) obj;
        if (((Boolean) obj2).booleanValue()) {
            mediaPlayer.start();
        }
    }
}