#include <Dynamixel2Arduino.h>

Dynamixel2Arduino dxl(Serial3, 84);
String incomingCommand = "";
unsigned long previousMillis = 0;

void setup() {
  Serial.begin(115200); 
  
  pinMode(BDPIN_DXL_PWR_EN, 1);
  digitalWrite(BDPIN_DXL_PWR_EN, 1);
  delay(500);

  dxl.begin(1000000);
  dxl.setPortProtocolVersion(2.0);

  dxl.torqueOff(1);
  dxl.torqueOff(2);
  dxl.setOperatingMode(1, OP_EXTENDED_POSITION); 
  dxl.setOperatingMode(2, OP_EXTENDED_POSITION);
  dxl.torqueOn(1);
  dxl.torqueOn(2);

  dxl.writeControlTableItem(ControlTableItem::PROFILE_VELOCITY, 1, 30);
  dxl.writeControlTableItem(ControlTableItem::PROFILE_VELOCITY, 2, 30);
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n') {
      int commaIndex = incomingCommand.indexOf(',');
      if (commaIndex > 0) {
        int leftSpeed = incomingCommand.substring(0, commaIndex).toInt();
        int rightSpeed = incomingCommand.substring(commaIndex + 1).toInt();
        dxl.setGoalPosition(1, leftSpeed, UNIT_DEGREE);
        dxl.setGoalPosition(2, rightSpeed, UNIT_DEGREE);
      }
      incomingCommand = "";
    } else if (c != '\r') {
      incomingCommand += c;
    }
  }

  if (millis() - previousMillis >= 50) {
    previousMillis = millis();
    Serial.print("E,");
    Serial.print(dxl.getPresentPosition(1, UNIT_DEGREE));
    Serial.print(",");
    Serial.println(dxl.getPresentPosition(2, UNIT_DEGREE));
  }
}