import React from 'react';
import { StyleSheet, Text, View, Button , Alert} from 'react-native';
import { ImagePicker, Permissions } from 'expo';


export default class Home extends React.Component {
  constructor(props) {
    super(props)
    this.state = {"image_data": null,hasCameraPermission: null,}
  }

  async componentWillMount() {
    let status_1  = Permissions.askAsync(Permissions.CAMERA);
    let status_2  = Permissions.askAsync(Permissions.CAMERA_ROLL);
    const results = await Promise.all([status_1, status_2]).then((status_1, status_2) => {
      //Alert.alert(JSON.stringify(status_1[0].status))
      //Alert.alert(status_2.status[0].status)
      this.setState({hasCameraPermission: status_1[0].status === 'granted'});
    })
  }

  GetImageData = async () => {
    let image = await ImagePicker.launchCameraAsync({
      allowsEditing: false,
      aspect: [4, 3],
      base64: true
    });
    this.setState({"image_data": image})
    Alert.alert("picture taken!")
  }


  render() {
    Alert.alert("test")
    const { hasCameraPermission } = this.state;
    if (typeof hasCameraPermission === null) {
      return <View />;
    } else if (hasCameraPermission === false) {
      return <Text>No access to camera</Text>;
    } else {
      return (
        <View style={styles.container}>
          <Text>Awesome Fruuittt</Text>
          <Button onPress={this.GetImageData} title="Take a picture" />
        </View>
      );
    }
  }
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
    alignItems: 'center',
    justifyContent: 'center',
  },
});
